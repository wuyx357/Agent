import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import os
import glob
import json
import math
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import faiss
from text2vec import SentenceModel
from enum import Enum
from typing import Dict, Any

load_dotenv()

# ==================== 配置 ====================
CHUNK_SIZE = 500
KNOWLEDGE_DIR = ".venv/knowledge"


# ==================== 输出格式枚举 ====================
class OutputFormat(str, Enum):
    NORMAL = "normal"  # 普通文本
    STRUCTURED = "structured"  # JSON 结构化
    CHAIN_OF_THOUGHT = "cot"  # 思维链
    CONCISE = "concise"  # 简洁模式


# ==================== Prompt 模板 ====================
class PromptTemplate:
    """Prompt 模板类"""

    # 普通问答模板
    QA_TEMPLATE = """你是一个专业的问答助手。请根据以下【参考知识】回答用户的问题。

【参考知识】
{context}

【对话历史】
{history}

【用户问题】
{query}

规则：
1. 只根据参考知识回答，不要编造
2. 如果参考知识不足以回答问题，请诚实说明
3. 回答要简洁、准确
4. 每次回答完都要说'滴'
"""

    # 结构化输出模板
    STRUCTURED_TEMPLATE = """你是一个专业的问答助手。请根据以下【参考知识】回答用户的问题，并以 JSON 格式输出。

【参考知识】
{context}

【用户问题】
{query}

输出格式要求：
{{
    "answer": "你的答案内容",
    "confidence": 0.0-1.0之间的数字（表示确定程度）,
    "sources": ["参考来源1", "参考来源2"],
    "has_answered": true,
    "reasoning": "你的推理过程"
}}

注意：只输出 JSON，不要输出其他内容。
"""

    # 思维链模板
    COT_TEMPLATE = """请一步一步思考，然后回答用户的问题。

【参考知识】
{context}

【用户问题】
{query}

请按以下格式输出：

## 思考过程
（逐步推理）

## 最终答案
（简洁回答）
"""

    # 工具调用模板（保留原有）
    TOOL_TEMPLATE = """你是一个智能助手，可以根据需要调用工具。

可用工具：
{tools_description}

用户问题：{query}

请判断是否需要调用工具。如果需要，输出 JSON 格式：
{
    "need_tool": true,
    "tool_name": "工具名称",
    "tool_params": {"参数名": "参数值"}
}

如果不需要调用工具，输出：
{
    "need_tool": false,
    "direct_answer": "直接回答"
}
"""


# ==================== 输出解析器 ====================
class OutputParser:
    """输出解析器"""

    @staticmethod
    def parse_structured_output(response: str) -> Dict[str, Any]:
        """解析结构化 JSON 输出"""
        try:
            # 尝试提取 JSON 内容
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response

            result = json.loads(json_str.strip())

            # 验证必要字段
            required_fields = ["answer", "confidence", "has_answered"]
            for field in required_fields:
                if field not in result:
                    result[field] = None

            return result
        except json.JSONDecodeError:
            return {
                "answer": response,
                "confidence": 0.5,
                "has_answered": True,
                "sources": [],
                "reasoning": "JSON 解析失败"
            }

    @staticmethod
    def parse_cot_output(response: str) -> Dict[str, str]:
        """解析思维链输出"""
        result = {"thinking": "", "answer": ""}

        if "## 思考过程" in response and "## 最终答案" in response:
            parts = response.split("## 最终答案")
            result["thinking"] = parts[0].replace("## 思考过程", "").strip()
            result["answer"] = parts[1].strip() if len(parts) > 1 else ""
        else:
            result["answer"] = response

        return result


# ==================== Prompt 构建器 ====================
class PromptBuilder:
    """Prompt 构建器"""

    def __init__(self):
        self.templates = PromptTemplate()

    def build_qa_prompt(self, query: str, context: str, history: list = None) -> str:
        """构建问答 Prompt"""
        history_text = self._format_history(history) if history else "无"
        return self.templates.QA_TEMPLATE.format(
            context=context,
            history=history_text,
            query=query
        )

    def build_structured_prompt(self, query: str, context: str) -> str:
        """构建结构化输出 Prompt"""
        return self.templates.STRUCTURED_TEMPLATE.format(
            context=context,
            query=query
        )

    def build_cot_prompt(self, query: str, context: str) -> str:
        """构建思维链 Prompt"""
        return self.templates.COT_TEMPLATE.format(
            context=context,
            query=query
        )

    def _format_history(self, history: list) -> str:
        """格式化对话历史"""
        formatted = []
        for msg in history[-6:]:  # 只保留最近6条
            role = "用户" if msg["role"] == "user" else "助手"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)


# ==================== 初始化 ====================
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

embedder = SentenceModel('shibing624/text2vec-base-chinese')

dimension = 768
index = faiss.IndexFlatL2(dimension)
documents_store = []

# ==================== 工具定义（保持不变） ====================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "时间格式，可选 'full'(完整) 或 'time'(仅时间)",
                        "enum": ["full", "time"]
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算，支持加减乘除、幂运算等",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2+3*4' 或 '2**10'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索企业内部知识库，获取相关文档内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的问题或关键词"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "发送一封电子邮件（模拟）",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "收件人邮箱地址"
                    },
                    "subject": {
                        "type": "string",
                        "description": "邮件主题"
                    },
                    "body": {
                        "type": "string",
                        "description": "邮件正文"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
]


# ==================== 工具实现（需要修复） ====================
def get_current_time(format_type="full"):
    now = datetime.now()
    if format_type == "time":
        return now.strftime("%H:%M:%S")
    else:
        return now.strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression):
    try:
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def search_knowledge(query, top_k=1):
    """搜索知识库（使用 FAISS）"""
    if len(documents_store) == 0:
        return "知识库为空，请先添加文档到 knowledge 文件夹"

    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(documents_store):
            results.append(documents_store[idx])

    if not results:
        return "未找到相关内容"

    return "\n\n---\n\n".join(results)


def send_email(to, subject, body):
    print(f"\n📧 模拟发送邮件:")
    print(f"   收件人: {to}")
    print(f"   主题: {subject}")
    print(f"   正文: {body[:100]}...")
    return f"邮件已发送至 {to}"


TOOL_HANDLERS = {
    "get_current_time": lambda args: get_current_time(args.get("format", "full")),
    "calculate": lambda args: calculate(args["expression"]),
    "search_knowledge": lambda args: search_knowledge(args["query"]),
    "send_email": lambda args: send_email(args["to"], args["subject"], args["body"])
}


# ==================== 知识库构建 ====================
def load_documents():
    documents = []
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        return documents
    for file_path in glob.glob(f"{KNOWLEDGE_DIR}/*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(content)
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE):
    chunks = []
    words = text.split('\n')
    current_chunk = ""
    for line in words:
        if len(current_chunk) + len(line) < chunk_size:
            current_chunk += line + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def build_knowledge_base():
    global index, documents_store
    print("📚 正在构建知识库...")
    documents = load_documents()

    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)

    if len(all_chunks) == 0:
        print("⚠️ 没有找到知识库文档")
        return

    print(f"📄 共 {len(all_chunks)} 个文本块，正在生成向量...")

    BATCH_SIZE = 100
    all_embeddings = []

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        embeddings = embedder.encode(batch)
        all_embeddings.extend(embeddings)
        print(f"   已处理 {min(i + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)}")

    embeddings_array = np.array(all_embeddings).astype('float32')

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    documents_store = all_chunks

    print(f"✅ 知识库构建完成，共 {len(all_chunks)} 个文本块")


def retrieve(query, top_k=3):
    if len(documents_store) == 0:
        return []
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(documents_store):
            results.append(documents_store[idx])
    return results


# ==================== 增强的生成答案（支持多种输出格式） ====================
def generate_answer_enhanced(query, context, output_format: OutputFormat = OutputFormat.NORMAL,
                             conversation_history: list = None):
    """增强的答案生成，支持多种输出格式"""

    prompt_builder = PromptBuilder()

    if output_format == OutputFormat.STRUCTURED:
        prompt = prompt_builder.build_structured_prompt(query, context)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return OutputParser.parse_structured_output(response.choices[0].message.content)

    elif output_format == OutputFormat.CHAIN_OF_THOUGHT:
        prompt = prompt_builder.build_cot_prompt(query, context)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return OutputParser.parse_cot_output(response.choices[0].message.content)

    else:
        # NORMAL 或 CONCISE 模式
        prompt = prompt_builder.build_qa_prompt(query, context, conversation_history)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return {"answer": response.choices[0].message.content}


# ==================== 主对话函数（支持模式切换） ====================
def chat_with_tools_enhanced():
    """增强的主对话函数，支持多种输出格式"""
    build_knowledge_base()

    messages = [
        {"role": "system", "content": "你是一个智能助手，可以使用工具来回答问题。每次回答完都要说'滴'。"}
    ]

    # 当前输出格式
    current_format = OutputFormat.NORMAL
    prompt_builder = PromptBuilder()

    print("\n" + "=" * 60)
    print("🤖 智能助手已启动（输入 'exit' 退出）")
    print("📌 支持功能：查时间、数学计算、知识库搜索、发邮件")
    print("")
    print("📌 格式切换命令：")
    print("   /format normal     - 普通模式")
    print("   /format structured - 结构化输出（JSON）")
    print("   /format cot        - 思维链模式")
    print("=" * 60 + "\n")

    while True:
        user_input = input("[用户]: ").strip()
        if user_input.lower() == "exit":
            print("👋 再见！")
            break

        # 处理格式切换命令
        if user_input.startswith("/format"):
            cmd = user_input.split()[1] if len(user_input.split()) > 1 else ""
            if cmd == "structured":
                current_format = OutputFormat.STRUCTURED
                print(f"✅ 已切换到【结构化输出】模式\n")
            elif cmd == "cot":
                current_format = OutputFormat.CHAIN_OF_THOUGHT
                print(f"✅ 已切换到【思维链】模式\n")
            elif cmd == "normal":
                current_format = OutputFormat.NORMAL
                print(f"✅ 已切换到【普通】模式\n")
            else:
                print(f"❌ 未知格式: {cmd}，可用: normal, structured, cot\n")
            continue

        messages.append({"role": "user", "content": user_input})

        # 第一次调用，让模型决定是否使用工具
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=False
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        # 如果有工具调用
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                print(f"🔧 [调用工具]: {function_name}({arguments})")

                handler = TOOL_HANDLERS.get(function_name)
                if handler:
                    result = handler(arguments)
                else:
                    result = f"错误：未知工具 {function_name}"

                print(f"📤 [工具返回]: {result[:200]}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            # 第二次调用，根据工具结果生成回答（也支持格式）
            second_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            final_answer = second_response.choices[0].message.content

            # 如果当前是结构化或思维链模式，尝试解析
            if current_format == OutputFormat.STRUCTURED:
                parsed = OutputParser.parse_structured_output(final_answer)
                print(f"\n[答案]: {parsed.get('answer', '')}")
                print(f"[置信度]: {parsed.get('confidence', 'N/A')}")
                if parsed.get('sources'):
                    print(f"[来源]: {parsed['sources'][0][:100]}...")
                print()
            elif current_format == OutputFormat.CHAIN_OF_THOUGHT:
                parsed = OutputParser.parse_cot_output(final_answer)
                if parsed.get('thinking'):
                    print(f"\n[思考过程]:\n{parsed['thinking']}")
                print(f"\n[答案]: {parsed['answer']}\n")
            else:
                print(f"[Agent]: {final_answer}\n")

            messages.append({"role": "assistant", "content": final_answer})
        else:
            # 没有工具调用，使用增强的答案生成
            # 先检索知识库
            retrieved_docs = retrieve(user_input)
            context = "\n\n---\n\n".join(retrieved_docs) if retrieved_docs else "无相关参考知识"

            # 根据当前格式生成答案
            if current_format == OutputFormat.STRUCTURED:
                result = generate_answer_enhanced(user_input, context, OutputFormat.STRUCTURED)
                print(f"\n[答案]: {result.get('answer', '')}")
                print(f"[置信度]: {result.get('confidence', 'N/A')}")
                print(f"[已回答]: {result.get('has_answered', False)}")
                if result.get('sources'):
                    print(f"[来源]: {result['sources'][0][:100]}...")
                if result.get('reasoning'):
                    print(f"[推理]: {result['reasoning'][:200]}...")
                print()
                final_answer = result.get('answer', '')

            elif current_format == OutputFormat.CHAIN_OF_THOUGHT:
                result = generate_answer_enhanced(user_input, context, OutputFormat.CHAIN_OF_THOUGHT)
                if result.get('thinking'):
                    print(f"\n[思考过程]:\n{result['thinking']}")
                print(f"\n[答案]: {result['answer']}\n")
                final_answer = result.get('answer', '')

            else:
                result = generate_answer_enhanced(user_input, context, OutputFormat.NORMAL, messages[:-1])
                final_answer = result.get('answer', '')
                print(f"[Agent]: {final_answer}\n")

            messages.append({"role": "assistant", "content": final_answer})


# ==================== 入口 ====================
if __name__ == "__main__":
    chat_with_tools_enhanced()