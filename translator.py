from pathlib import Path
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, TRANSLATED_DIR

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

TRANSLATION_PROMPT = """你是一个技术文档翻译专家。请将以下英文技术文档翻译成中文。

要求：
1. 保持技术术语的准确性（如 Agent、Tool Use、Prompt Engineering 等保留英文原词或使用业界通用译法）
2. 代码块保持原样不翻译
3. Markdown 格式（标题、列表、链接、代码块）完整保留
4. 翻译结果流畅自然，符合中文技术文档的阅读习惯

英文原文：

{text}

中文翻译："""


def translate_text(text: str) -> str:
    """翻译单段文本，保留 Markdown 格式。"""
    if not text.strip():
        return text

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": TRANSLATION_PROMPT.format(text=text)}],
        temperature=0.3,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def translate_file(file_path: Path, output_dir: str | None = None) -> Path:
    """翻译单个 Markdown 文件，保存到输出目录。"""
    output = Path(output_dir or TRANSLATED_DIR)
    output.mkdir(parents=True, exist_ok=True)

    content = file_path.read_text(encoding="utf-8")
    translated = translate_text(content)

    out_path = output / file_path.name
    out_path.write_text(translated, encoding="utf-8")

    return out_path


def translate_directory(input_dir: str | None = None, output_dir: str | None = None) -> list[Path]:
    """翻译目录下所有 Markdown 文件。"""
    input_path = Path(input_dir)
    output_path = Path(output_dir or TRANSLATED_DIR)

    md_files = list(input_path.rglob("*.md"))
    results = []

    for f in md_files:
        out = translate_file(f, output_dir=str(output_path))
        results.append(out)
        print(f"  翻译完成: {f.name} → {out.name}")

    return results
