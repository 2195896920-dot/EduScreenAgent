# screen_agent/core/memory.py
import tiktoken
from typing import List, Dict, Any
from core.llm import call_llm_simple


class Memory:
    """对话记忆管理：超长时自动压缩旧对话"""

    def __init__(self, max_tokens: int = 4000, keep_recent_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.summary = ""

        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        if self.encoding and text:
            return len(self.encoding.encode(text))
        return len(text)  # 粗略估算

    def count_messages_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "") or ""
            total += self.count_tokens(content)
        return total

    def process(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        检查 messages 的 token 数，如果超了，总结旧对话。
        注意：不修改原列表，返回新列表。
        """
        total_tokens = self.count_messages_tokens(messages)

        if total_tokens <= self.max_tokens:
            return messages

        # 从后往前算，保留 keep_recent_tokens 的内容
        keep_count = 0
        keep_tokens = 0
        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = self.count_tokens(messages[i].get("content", ""))
            if keep_tokens + msg_tokens > self.keep_recent_tokens:
                break
            keep_tokens += msg_tokens
            keep_count += 1

        keep_count = max(keep_count, 2)  # 至少保留一轮对话
        to_summarize = messages[:-keep_count]
        keep_messages = messages[-keep_count:]

        # 拼成文字，让 LLM 总结
        dialog_text = ""
        for msg in to_summarize:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "tool" and len(content) > 200:
                content = content[:200] + "..."
            dialog_text += f"{role}: {content}\n"

        prompt = f"""请总结以下对话的关键信息，保留用户的重要需求、背景、偏好。用简洁的中文：

{dialog_text}

总结："""

        new_summary = call_llm_simple(prompt)

        if self.summary:
            self.summary = f"{self.summary}\n{new_summary}"
        else:
            self.summary = new_summary

        result = []
        if self.summary:
            result.append({
                "role": "system",
                "content": f"以下是之前对话的总结：{self.summary}"
            })
        result.extend(keep_messages)
        return result