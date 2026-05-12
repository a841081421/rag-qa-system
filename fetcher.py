import os
import subprocess
from pathlib import Path
from config import KNOWLEDGE_BASE_DIR

REPOS = [
    "https://github.com/anthropics/anthropic-cookbook.git",
    "https://github.com/anthropics/courses.git",
]


def clone_repos(target_dir: str | None = None) -> list[Path]:
    """克隆 Anthropic 开源仓库到目标目录，返回所有 Markdown 文件路径列表。"""
    target = Path(target_dir or KNOWLEDGE_BASE_DIR)
    target.mkdir(parents=True, exist_ok=True)

    md_files = []

    for repo_url in REPOS:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        repo_path = target / repo_name

        if repo_path.exists():
            subprocess.run(["git", "-C", str(repo_path), "pull"], check=True)
        else:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(repo_path)], check=True
            )

        md_files.extend(list(repo_path.rglob("*.md")))

    return md_files


def list_markdown_files(base_dir: str | None = None) -> list[Path]:
    """列出 knowledge_base 下所有 Markdown 文件。"""
    base = Path(base_dir or KNOWLEDGE_BASE_DIR)
    if not base.exists():
        return []
    return list(base.rglob("*.md"))
