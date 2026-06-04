"""LLM generator with streaming, conversation history, and JSON mode."""

from __future__ import annotations

import json
import re
from typing import Generator

from openai import OpenAI, APIError
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

SYSTEM_PROMPT = """你是一个 AI 技术知识助手。请基于提供的参考资料回答用户问题。

规则：
1. 如果参考资料中有答案，基于资料回答，并在末尾标注引用的来源
2. 如果参考资料不包含相关信息，如实告知用户，不要编造
3. 回答要结构化、清晰，适合技术读者阅读
4. 涉及代码时，给出可运行的示例
5. 综合多个来源时，对比分析并给出完整视图"""

CONTEXT_PROMPT_TEMPLATE = """参考资料：
{context}

用户问题：{query}

请基于以上参考资料回答问题。如果对话历史中有相关上下文，结合历史信息一并回答。"""

JSON_SYSTEM_PROMPT = """你是一个 AI 技术知识助手。你必须以 JSON 格式回答。

严格返回如下 JSON 结构（不要包含 markdown 代码块标记）：
{{
  "answer": "详细回答内容",
  "confidence": 0.95,
  "key_points": ["要点1", "要点2"],
  "sources": ["来源1", "来源2"]
}}"""


def _build_messages(
    query: str,
    contexts: list[str],
    history: list[dict] | None = None,
    json_mode: bool = False,
) -> list[dict]:
    system = JSON_SYSTEM_PROMPT if json_mode else SYSTEM_PROMPT
    messages = [{"role": "system", "content": system}]

    if history:
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    context_text = "\n\n---\n\n".join(
        f"[来源 {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )
    user_content = CONTEXT_PROMPT_TEMPLATE.format(context=context_text, query=query)
    messages.append({"role": "user", "content": user_content})
    return messages


class Generator:
    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def generate(
        self,
        query: str,
        contexts: list[str],
        history: list[dict] | None = None,
    ) -> str:
        messages = _build_messages(query, contexts, history)
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
            )
        except APIError as e:
            raise RuntimeError(f"LLM API 调用失败: {e}") from e
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("模型返回了空内容")
        return content

    def generate_stream(
        self,
        query: str,
        contexts: list[str],
        history: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        messages = _build_messages(query, contexts, history)
        try:
            stream = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except APIError as e:
            raise RuntimeError(f"LLM API 调用失败: {e}") from e

    def generate_json(
        self,
        query: str,
        contexts: list[str],
        history: list[dict] | None = None,
    ) -> dict:
        messages = _build_messages(query, contexts, history, json_mode=True)
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
            )
        except APIError as e:
            raise RuntimeError(f"LLM API 调用失败: {e}") from e
        content = response.choices[0].message.content or ""
        content = re.sub(r'^```\w*\n?', '', content.strip())
        content = content.rstrip('`').strip()
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {
                "answer": content,
                "confidence": 0.0,
                "key_points": [],
                "sources": [],
            }
