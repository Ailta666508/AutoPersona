"""Verify research-artifact exclusions, release source hashes, and syntax."""

import ast
import hashlib
import json
from pathlib import Path


PREPRINT_PATH = Path("paper/AutoPersona_Preprint.pdf")
PREPRINT_SHA256 = "963bebcd4e6fce6c5bbeded9eb6d4f4d1d3cb2e715d2d5a921aae19218686661"
PAPER_FIGURE_SHA256 = {
    Path("assets/figures/figure-1-motivation.png"): "4d6f295f3b64d80362e017e094c985ce6b0bd356aa1e75b163b2aafef7c369c0",
    Path("assets/figures/figure-2-framework.png"): "e6aa012e152b4066b5aaded040d09c1f18d7cc8b99cbe5ae7bd45528dd076807",
    Path("assets/figures/figure-3-clarification-analysis.png"): "21116aadfa76599a398d4993f3bd775735cdadb43214050c6c016926b2c88ec2",
}


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

    preprint = root / PREPRINT_PATH
    if not preprint.is_file():
        errors.append(f"Missing public preprint: {PREPRINT_PATH}")
    elif hashlib.sha256(preprint.read_bytes()).hexdigest() != PREPRINT_SHA256:
        errors.append(f"Public preprint checksum changed: {PREPRINT_PATH}")

    for figure_path, expected_sha256 in PAPER_FIGURE_SHA256.items():
        figure = root / figure_path
        if not figure.is_file():
            errors.append(f"Missing paper figure: {figure_path}")
        elif hashlib.sha256(figure.read_bytes()).hexdigest() != expected_sha256:
            errors.append(f"Paper figure checksum changed: {figure_path}")

    forbidden_roots = {"verl-0.7.0", "checkpoints", "results"}
    forbidden_suffixes = {".doc", ".docx"}
    for path in root.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        rel = path.relative_to(root)
        unexpected_pdf = path.suffix.lower() == ".pdf" and rel != PREPRINT_PATH
        if (
            rel.parts[0] in forbidden_roots
            or path.suffix.lower() in forbidden_suffixes
            or unexpected_pdf
        ):
            errors.append(f"Excluded artifact present: {rel}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"Verified {len(manifest['files'])} core files, the public preprint, "
        f"{len(PAPER_FIGURE_SHA256)} paper figures, and research-artifact exclusions."
    )


if __name__ == "__main__":
    main()
