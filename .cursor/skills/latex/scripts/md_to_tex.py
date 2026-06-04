#!/usr/bin/env python
"""
Convert Markdown report to Overleaf-ready LaTeX (main.tex + figures/).
Does not compile PDF.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

LATEX_SPECIALS = str.maketrans(
    {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
)

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}

\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage[margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{amsmath,amssymb}
\usepackage{siunitx}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{enumitem}

\hypersetup{hidelinks}
\captionsetup{font=small,labelfont=bf}
\graphicspath{{figures/}}

\title{%TITLE%}
\author{Adult Income ML Study}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage

"""


def escape_latex(text: str) -> str:
    return text.translate(LATEX_SPECIALS)


def inline_format(text: str) -> str:
    placeholders: list[str] = []

    def _ph(cmd: str, inner: str) -> str:
        token = f"@@LATEX{len(placeholders)}@@"
        placeholders.append(f"\\{cmd}{{{escape_latex(inner)}}}")
        return token

    text = re.sub(r"\*\*(.+?)\*\*", lambda m: _ph("textbf", m.group(1)), text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", lambda m: _ph("textit", m.group(1)), text)
    text = re.sub(r"`(.+?)`", lambda m: _ph("texttt", m.group(1)), text)
    text = escape_latex(text)
    for i, repl in enumerate(placeholders):
        text = text.replace(f"@@LATEX{i}@@", repl)
    return text


def normalize_figure_path(path: str) -> str:
    p = path.replace("\\", "/").strip()
    for prefix in ("./", "reports/"):
        if p.startswith(prefix):
            p = p[len(prefix) :]
    if p.startswith("figures/"):
        return p
    if "/figures/" in p:
        return "figures/" + p.split("/figures/")[-1]
    return f"figures/{Path(p).name}"


def parse_table(lines: list[str]) -> str:
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return ""
    ncol = len(rows[0])
    colspec = "l" * ncol
    out = [
        "\\begin{table}[H]",
        "  \\centering",
        "  \\small",
        f"  \\begin{{tabular}}{{{colspec}}}",
        "    \\toprule",
    ]
    for i, row in enumerate(rows):
        padded = row + [""] * (ncol - len(row))
        line = " & ".join(inline_format(c) for c in padded[:ncol]) + r" \\"
        out.append("    " + line)
        if i == 0:
            out.append("    \\midrule")
    out.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
    return "\n".join(out)


def convert_md_to_latex(md_path: Path, out_dir: Path) -> Path:
    text = md_path.read_text(encoding="utf-8")
    md_dir = md_path.parent
    out_fig = out_dir / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    lines = text.splitlines()
    body: list[str] = []
    i = 0
    title = "Report"
    in_table = False
    table_buf: list[str] = []

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("|") and not in_table:
            in_table = True
            table_buf = [line]
            i += 1
            continue

        if in_table:
            if line.strip().startswith("|"):
                table_buf.append(line)
                i += 1
                continue
            body.append(parse_table(table_buf))
            in_table = False
            table_buf = []
            continue

        if line.startswith("# ") and not line.startswith("## "):
            title = line[2:].strip()
            i += 1
            continue

        m_img = re.match(r"^!\[(.*?)\]\((.+?)\)$", line.strip())
        if m_img:
            alt, src = m_img.group(1), m_img.group(2)
            rel = normalize_figure_path(src)
            src_norm = src.replace("\\", "/").lstrip("./")
            src_path = md_dir / src_norm
            if not src_path.exists():
                src_path = md_dir / rel
            if not src_path.exists():
                src_path = md_dir / "figures" / Path(src_norm).name
            if src_path.exists():
                dest = out_fig / Path(rel).name
                shutil.copy2(src_path, dest)
            else:
                body.append(f"% MISSING IMAGE: {rel}")
            cap = inline_format(alt) if alt else "Figure"
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("*") and lines[i + 1].strip().endswith("*"):
                cap = inline_format(lines[i + 1].strip().strip("*"))
                i += 1
            body.extend(
                [
                    "\\begin{figure}[H]",
                    "  \\centering",
                    f"  \\includegraphics[width=0.9\\linewidth]{{{rel}}}",
                    f"  \\caption{{{cap}}}",
                    "\\end{figure}",
                    "",
                ]
            )
            i += 1
            continue

        if line.startswith("## "):
            sec = re.sub(r"^\d+\s+", "", line[3:].strip())
            body.append(f"\\section{{{inline_format(sec)}}}")
            body.append("")
            i += 1
            continue

        if line.startswith("### "):
            sub = re.sub(r"^\d+\.\d+\s+", "", line[4:].strip())
            body.append(f"\\subsection{{{inline_format(sub)}}}")
            body.append("")
            i += 1
            continue

        if line.strip() == "---":
            body.append("")
            i += 1
            continue

        if line.strip().startswith(">"):
            quote = line.strip().lstrip(">").strip()
            body.append("\\begin{quotation}")
            body.append(inline_format(quote))
            body.append("\\end{quotation}")
            body.append("")
            i += 1
            continue

        if re.match(r"^\d+\.\s", line.strip()):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            body.append("\\begin{enumerate}")
            for it in items:
                body.append(f"  \\item {inline_format(it)}")
            body.append("\\end{enumerate}")
            body.append("")
            continue

        if line.strip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:])
                i += 1
            body.append("\\begin{itemize}")
            for it in items:
                body.append(f"  \\item {inline_format(it)}")
            body.append("\\end{itemize}")
            body.append("")
            continue

        if line.strip().startswith("*") and line.strip().endswith("*") and not line.strip().startswith("**"):
            body.append(inline_format(line.strip().strip("*")))
            body.append("")
            i += 1
            continue

        if line.strip():
            body.append(inline_format(line.strip()))
            body.append("")
        i += 1

    if in_table and table_buf:
        body.append(parse_table(table_buf))

    bib_src = md_path.parent / "references.bib"
    bib_block = ""
    if bib_src.exists():
        shutil.copy2(bib_src, out_dir / "references.bib")
        bib_block = "\n\\bibliographystyle{plain}\n\\bibliography{references}\n"

    main = (
        PREAMBLE.replace("%TITLE%", escape_latex(title))
        + "\n".join(body)
        + bib_block
        + "\n\\end{document}\n"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    main_path = out_dir / "main.tex"
    main_path.write_text(main, encoding="utf-8")
    return main_path


def main():
    parser = argparse.ArgumentParser(description="Markdown to Overleaf LaTeX (no PDF)")
    parser.add_argument("markdown", type=Path, help="Source .md file")
    parser.add_argument("-o", "--output", type=Path, default=Path("reports/overleaf"))
    args = parser.parse_args()
    out = convert_md_to_latex(args.markdown, args.output)
    print(f"Wrote {out}")
    print(f"Figures: {args.output / 'figures'}")
    print("Upload reports/overleaf/ to Overleaf. Do not compile PDF locally unless requested.")


if __name__ == "__main__":
    main()
