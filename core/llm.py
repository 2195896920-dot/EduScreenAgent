import os
import json
from typing import Optional, List, Dict, Any, AsyncGenerator
from openai import OpenAI, AsyncOpenAI


# ================== 客户端初始化 ==================
def _get_client() -> OpenAI:
    """同步客户端（用于普通调用、工具调用、JSON模式）"""
    api_key = os.environ.get("KIMICODE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("KIMICODE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("环境变量 KIMICODE_API_KEY 或 OPENAI_API_KEY 未设置")
    
    # ⚠️ Kimi For Coding 要求特定 User-Agent，否则 403
    # 默认用 KimiCLI/1.5，可通过环境变量 USER_AGENT 覆盖
    user_agent = os.environ.get("USER_AGENT", "KimiCLI/1.5")
    
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"User-Agent": user_agent},
    )


def _get_async_client() -> AsyncOpenAI:
    """异步客户端（仅用于流式调用）"""
    api_key = os.environ.get("KIMICODE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("KIMICODE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("环境变量 KIMICODE_API_KEY 或 OPENAI_API_KEY 未设置")
    
    user_agent = os.environ.get("USER_AGENT", "KimiCLI/1.5")
    
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"User-Agent": user_agent},
    )


def _get_model() -> str:
    return os.environ.get("KIMICODE_MODEL") or os.environ.get("OPENAI_MODEL") or "kimi-k2.6"


# ================== 基础调用（兼容原有） ==================
def call_llm_simple(prompt: str) -> str:
    """最简单的方式：给一句话，返回一句话"""
    client = _get_client()
    model = _get_model()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def call_llm(
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict]] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """完整方式：支持多轮对话和工具调用（兼容原有 TianrunClaw）"""
    msgs = list(messages)
    if system_prompt:
        msgs = [{"role": "system", "content": system_prompt}, *msgs]

    client = _get_client()
    model = _get_model()

    kwargs = {"model": model, "messages": msgs}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    reasoning_content = getattr(message, "reasoning_content", None)

    result = {
        "role": "assistant",
        "content": message.content or "",
        "reasoning_content": reasoning_content or "",
    }

    if message.tool_calls:
        result["tool_calls"] = [
            tool_call.model_dump() for tool_call in message.tool_calls
        ]

    return result


# ================== 流式调用（异步生成器） ==================
async def call_llm_stream(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    流式调用 LLM，返回异步文本生成器。
    用法：async for chunk in call_llm_stream(...): ...
    """
    msgs = list(messages)
    if system_prompt:
        msgs = [{"role": "system", "content": system_prompt}, *msgs]

    client = _get_async_client()
    model = _get_model()

    response = await client.chat.completions.create(
        model=model,
        messages=msgs,
        stream=True,
    )

    async for chunk in response:
        if not chunk.choices:  # ← 加这行：跳过空 choices 的 chunk
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# ================== 强制 JSON 输出（结构化数据） ==================
def call_llm_json(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    json_schema: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    强制 LLM 返回合法 JSON。
    用于：根据量表得分生成分级结果、生成 IEP 结构化字段等。
    """
    msgs = list(messages)
    if system_prompt:
        msgs = [{"role": "system", "content": system_prompt}, *msgs]

    client = _get_client()
    model = _get_model()

    extra = {}
    if json_schema:
        extra["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": json_schema,
                "strict": True,
            }
        }
    else:
        extra["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(
        model=model,
        messages=msgs,
        **extra,
    )
    
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "LLM 返回了非法 JSON", "raw": content}


# ================== 评估报告专用生成 ==================
def generate_report(
    assessment_data: Dict[str, Any],
    system_prompt: Optional[str] = None,
) -> str:
    """
    根据评估数据生成标准化报告。
    assessment_data 包含：儿童信息、量表得分、风险等级、RAG检索结果等。
    """
    default_system = (
        "你是一位特殊教育评估专家，正在为南宁市爱华小学撰写"
        "《特需儿童入学能力评估报告》。报告要求：\n"
        "1. 语言专业、客观、温暖，避免歧视性表述\n"
        "2. 结构清晰：评估背景→量表结果→风险分析→分班建议→干预方案\n"
        "3. 所有建议必须基于提供的学校资源数据，不要虚构\n"
        "4. 使用中文，医学术语后附通俗解释\n"
    )
    
    prompt = f"""请根据以下评估数据生成正式报告：

【儿童基本信息】
{json.dumps(assessment_data.get("child", {}), ensure_ascii=False, indent=2)}

【量表得分】
{json.dumps(assessment_data.get("scores", {}), ensure_ascii=False, indent=2)}

【风险等级】
{assessment_data.get("risk_level", "未知")}

【学校资源匹配结果】
{assessment_data.get("rag_result", "无")}

请生成完整的 Markdown 格式评估报告。
"""

    client = _get_client()
    model = _get_model()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt or default_system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""