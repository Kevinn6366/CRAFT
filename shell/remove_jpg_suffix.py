#!/usr/bin/env python3
"""Strip extensions from file paths in text files, keeping only basename stems."""

import argparse
from pathlib import Path


def process_file(path: Path, make_backup: bool = True) -> None:
    if not path.exists():
        print(f"File not found: {path}")
        return

    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    new_lines = []
    for ln in lines:
        try:
            name = Path(ln).stem
        except Exception:
            name = ln
        new_lines.append(name)

    if make_backup:
        backup_path = path.with_name(path.name + ".bak")
        backup_path.write_text(text, encoding="utf-8")

    path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
    print(f"Processed {path}: {len(new_lines)} lines, backup={make_backup}")


def main():
    parser = argparse.ArgumentParser(description="Strip extensions from file paths in list files.")
    parser.add_argument("files", nargs="+", help="Text files to process")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup files")
    args = parser.parse_args()

    for f in args.files:
        process_file(Path(f), make_backup=not args.no_backup)


if __name__ == "__main__":
    main()
