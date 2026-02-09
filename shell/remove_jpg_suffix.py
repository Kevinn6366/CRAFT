#!/usr/bin/env python3
"""
从给定的文本文件中去除行中路径与扩展名，只保留文件名（不含扩展名）。
会为每个处理的文件生成一个备份（同名加 .bak）。
用法示例：
python3 shell/remove_jpg_suffix.py path/to/test.txt path/to/train.txt path/to/val.txt
"""

import argparse
from pathlib import Path


def process_file(path: Path, make_backup: bool = True) -> None:
    if not path.exists():
        print(f"文件不存在: {path}")
        return

    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()]
    # 过滤空行，同时保留顺序
    lines = [ln for ln in lines if ln]

    new_lines = []
    for ln in lines:
        # 可能行是完整路径、相对路径或仅文件名；取 basename 的 stem（不含扩展名）
        try:
            name = Path(ln).stem
        except Exception:
            name = ln
        new_lines.append(name)

    if make_backup:
        backup_path = path.with_name(path.name + ".bak")
        backup_path.write_text(text, encoding="utf-8")

    path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
    print(f"已处理 {path}，共 {len(new_lines)} 行，备份: {make_backup}")


def main():
    parser = argparse.ArgumentParser(description="从列表文件中去掉扩展名，只保留文件名（无扩展名）。")
    parser.add_argument("files", nargs="+", help="要处理的文本文件路径（可以传多个）。")
    parser.add_argument("--no-backup", action="store_true", help="不创建备份文件")
    args = parser.parse_args()

    for f in args.files:
        process_file(Path(f), make_backup=not args.no_backup)


if __name__ == "__main__":
    main()
