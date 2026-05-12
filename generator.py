from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

SYSTEM_PROMPT = """你是一个 AI 技术知识助手。请基于提供的参考资料回答用户问题。

规则：
1. 如果参考资料中有答案，基于资料回答，并在末尾标注引用的来源
2. 如果参考资料不包含相关信息，如实告知用户，不要编造
3. 回答要结构化、清晰，适合技术读者阅读
4. 涉及代码时，给出可运行的示例"""


class Generator:
    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def generate(self, query: str, contexts: list[str]) -> str:
        """基于检索到的上下文生成回答。"""
        context_text = "\n\n---\n\n".join(
            f"[来源 {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )

        user_prompt = f"""参考资料：
{context_text}

用户问题：{query}

请基于以上参考资料回答问题。"""

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        return response.choices[0].message.content
