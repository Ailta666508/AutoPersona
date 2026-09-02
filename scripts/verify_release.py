"""Verify research-artifact exclusions, release source hashes, and syntax."""

import ast
import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/source-manifest.json").read_text())
    errors = []
    for entry in manifest["files"]:
        path = root / entry["release_path"]
        if not path.is_file():
            errors.append(f"Missing source: {entry['release_path']}")
            continue
        content = path.read_bytes()
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            errors.append(f"Original source changed: {entry['release_path']}")
        try:
            ast.parse(content, filename=entry["release_path"])
        except SyntaxError as error:
            errors.append(f"Syntax error: {entry['release_path']}: {error}")

    forbidden_roots = {"verl-0.7.0", "checkpoints", "results"}
    forbidden_suffixes = {".pdf", ".doc", ".docx"}
    for path in root.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts[0] in forbidden_roots or path.suffix.lower() in forbidden_suffixes:
            errors.append(f"Excluded artifact present: {rel}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(manifest['files'])} core files and research-artifact exclusions.")


if __name__ == "__main__":
    main()
