---
name: latex
description: >-
  Converts Markdown reports to LaTeX for Overleaf upload. Handles figures,
  booktabs tables, fonts, and section structure. Use when the user invokes /latex,
  asks for Overleaf, md2tex, or LaTeX export from report.md. Does not compile PDF.
disable-model-invocation: true
---

# /latex — Markdown to Overleaf LaTeX

Converts project Markdown (especially `reports/report.md`) into an **Overleaf-ready** `.tex` project: `main.tex`, optional `references.bib`, and a `figures/` folder with correct paths. **Do not run `pdflatex` or export PDF** unless the user explicitly asks later.

## When to use

- User types `/latex` or asks to convert a report to LaTeX / Overleaf.
- After `reports/report.md` is complete and figures exist under `reports/figures/`.

## Workflow

1. **Confirm inputs**
   - Source: default `reports/report.md`.
   - Figures dir: default `reports/figures/` (PNG preferred).

2. **Run the converter** (preferred):

```powershell
python .cursor/skills/latex/scripts/md_to_tex.py reports/report.md -o reports/overleaf
```

3. **Verify output** (checklist below). Fix paths or tables manually if needed.

4. **Tell the user how to upload to Overleaf**
   - New Project → Upload Project → zip `reports/overleaf/` (or upload `main.tex` + `figures/` + `references.bib`).
   - Set compiler to **pdfLaTeX** or **LuaLaTeX** (script targets pdfLaTeX + T1 fonts).
   - Click Recompile in Overleaf (user-side; agent does not compile locally).

5. **Do not** run `pdflatex`, `latexmk`, or produce PDF in the repo unless requested.

## Output layout (Overleaf project)

```
reports/overleaf/
├── main.tex          # \documentclass + body
├── references.bib    # copied if reports/references.bib exists
└── figures/          # copies of PNGs referenced in the report
```

Image paths in `main.tex` use `figures/filename.png` (no `./`, no `reports/` prefix).

## LaTeX conventions (this skill)

| Element | LaTeX approach |
|---------|----------------|
| Font | `\usepackage[T1]{fontenc}`, `\usepackage{lmodern}`, `\usepackage{microtype}` |
| Math | `amsmath`, `amssymb` |
| Figures | `graphicx`, `float`, `caption`; `[width=\linewidth]` or `0.85\linewidth` |
| Tables | `booktabs`, `tabularx`; header row `\toprule`, `\midrule`, `\bottomrule` |
| Links | `hyperref` with `hidelinks` |
| Sections | `\section`, `\subsection` from `#` / `##` |
| Bold / italic | `\textbf{}`, `\textit{}` |
| Lists | `enumerate` / `itemize` |

## Markdown mapping rules

- `# Title` → `\title{}` + `\maketitle` (first H1 only).
- `## N Section` → `\section{Section}` (strip leading number if duplicated).
- `###` → `\subsection{}`.
- `![caption](path)` → `figure` environment + `\includegraphics{figures/...}`; copy PNG to `overleaf/figures/`.
- `*caption*` line after image → `\caption{}`.
- Markdown pipe tables → `booktabs` `tabular` (see script).
- `` `code` `` → `\texttt{}`.
- `**bold**` → `\textbf{}`.
- Escape LaTeX specials in prose: `\` `&` `%` `$` `#` `_` `{` `}` `~` `^`.

## Image path fixes

| Markdown source | LaTeX path |
|-----------------|------------|
| `./figures/foo.png` | `figures/foo.png` |
| `figures/foo.png` | `figures/foo.png` |
| `reports/figures/foo.png` | `figures/foo.png` |

If an image is missing on disk, warn the user and insert `% MISSING: figures/foo.png` — do not invent placeholders.

## Table quality bar

- Use `booktabs` only (no vertical rules).
- Left-align text columns; `S` column type from `siunitx` for numeric columns when appropriate.
- For wide tables: `tabularx` with `\textwidth` or `longtable` + `\small`.
- Caption above table for tables in formal sections if report used “Table N” prose.

## Verification before handoff

- [ ] Every `\includegraphics` file exists in `overleaf/figures/`.
- [ ] No remaining `![](` or raw Markdown table syntax in `main.tex`.
- [ ] Special characters escaped in body text.
- [ ] Bibliography: `\bibliography{references}` if `.bib` copied.
- [ ] `main.tex` compiles in principle (no unclosed environments); user compiles on Overleaf.

## Manual fallback

If the script fails, read [reference.md](reference.md) and convert section-by-section using the preamble template there.

## Project-specific default

For this repo, default source is `reports/report.md` and figures from `reports/figures/`. Output to `reports/overleaf/`.
