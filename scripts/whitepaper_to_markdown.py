#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Regenerate doc/sparse_attention_whitepaper.md from the LaTeX source, so the
Markdown copy linked from the README never drifts from the paper.

    python scripts/whitepaper_to_markdown.py

Requires pandoc. Citations become numbered references ([1], [2], ...) in
\\bibitem order, and math is written as $...$ / $$...$$ for GitHub.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "doc" / "sparse_attention_whitepaper.tex"
TARGET = ROOT / "doc" / "sparse_attention_whitepaper.md"

PANDOC_FORMAT = "markdown" + "".join(f"-{ext}" for ext in (
    "header_attributes", "fenced_divs", "native_divs", "bracketed_spans", "native_spans",
    "raw_tex", "smart", "auto_identifiers", "implicit_figures", "link_attributes"))


def command_argument(tex: str, name: str) -> str:
    match = re.search(r"\\" + name + r"\{(.*?)\}", tex, re.S)
    if not match:
        sys.exit(f"could not find \\{name} in {SOURCE}")
    return match.group(1).strip()


def to_markdown(latex: str) -> str:
    result = subprocess.run(
        ["pandoc", "-f", "latex", "-t", PANDOC_FORMAT, "--wrap=none", "--shift-heading-level-by=1"],
        input=latex, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def display_math_on_own_lines(markdown: str) -> str:
    """Write display math as fenced ```math blocks, indented like the
    surrounding block (e.g. inside a list item). GitHub renders fenced math
    literally; inside $$...$$ its Markdown pass eats escapes such as \\{."""
    out = []
    for paragraph in markdown.split("\n\n"):
        if "$$" not in paragraph:
            out.append(paragraph)
            continue
        indent = re.match(r"\s*", paragraph).group(0)
        if re.match(r"\s*\d+\.\s", paragraph):  # list item: continuation indent
            indent = " " * len(re.match(r"\s*\d+\.\s+", paragraph).group(0))
        pieces = re.split(r"\$\$(.+?)\$\$", paragraph, flags=re.S)
        blocks = []
        for n, piece in enumerate(pieces):
            if n % 2:
                math = " ".join(line.strip() for line in piece.strip().splitlines())
                blocks.append(f"{indent}```math\n{indent}{math}\n{indent}```")
            elif piece.strip():
                blocks.append(piece.rstrip() if n == 0 else indent + piece.strip())
        out.append("\n\n".join(blocks))
    return "\n\n".join(out)


def protect_inline_math(markdown: str) -> str:
    """Use GitHub's literal $`...`$ form for inline math containing escaped
    punctuation (\\{, \\,, \\; ...), which plain $...$ would lose."""
    def fix(match):
        math = match.group(1)
        return f"$`{math}`$" if re.search(r"\\[^A-Za-z]", math) else match.group(0)
    return re.sub(r"(?<![$`])\$(?![$`])(.+?)(?<![$`])\$(?![$`])", fix, markdown)


def main():
    if shutil.which("pandoc") is None:
        sys.exit("pandoc is required: https://pandoc.org/installing.html")
    tex = SOURCE.read_text()

    title = command_argument(tex, "title")
    author = command_argument(tex, "author")
    date = command_argument(tex, "date")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
    body = re.search(r"\\maketitle(.*?)\\end\{document\}", tex, re.S).group(1)
    body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", body, flags=re.S)

    # Number citations in bibliography order
    keys = re.findall(r"\\bibitem\{([^}]+)\}", body)
    numbers = {key: i + 1 for i, key in enumerate(keys)}

    def cite(match):
        cited = [k.strip() for k in match.group(1).split(",")]
        missing = [k for k in cited if k not in numbers]
        if missing:
            sys.exit(f"citation without \\bibitem: {', '.join(missing)}")
        return " [" + ", ".join(str(numbers[k]) for k in cited) + "]"

    body = re.sub(r"~?\\cite\{([^}]+)\}", cite, body)
    body = re.sub(r"\\begin\{thebibliography\}\{[^}]*\}",
                  r"\\section*{References}\n\\begin{enumerate}", body)
    body = body.replace(r"\end{thebibliography}", r"\end{enumerate}")
    body = re.sub(r"\\bibitem\{[^}]+\}", r"\\item", body)
    # quote environments only indent the definitions in print; as Markdown
    # blockquotes they break display math
    body = re.sub(r"\\(begin|end)\{quote\}", "", body)

    body_md = to_markdown(body)
    # pandoc escapes the literal brackets of "[1]"; they need no escaping on GitHub
    body_md = re.sub(r"\\\[(\d+(?:, \d+)*)\\\]", r"[\1]", body_md)
    body_md = protect_inline_math(display_math_on_own_lines(body_md))

    markdown = "\n\n".join([
        f"# {to_markdown(title)}",
        f"**Author:** {to_markdown(author)}  \n**Date:** {to_markdown(date)}",
        "<!-- Generated from sparse_attention_whitepaper.tex by "
        "scripts/whitepaper_to_markdown.py. Edit the .tex and rerun the script. -->",
        "---",
        "## Abstract",
        protect_inline_math(to_markdown(abstract)),
        "---",
        body_md,
    ]) + "\n"
    TARGET.write_text(markdown)
    print(f"wrote {TARGET.relative_to(ROOT)} ({len(keys)} references)")


if __name__ == "__main__":
    main()
