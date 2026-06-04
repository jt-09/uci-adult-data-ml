# LaTeX reference — Overleaf preamble template

Use this preamble in `main.tex` when building by hand:

```latex
\documentclass[11pt,a4paper]{article}

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

\title{Predicting High Income from Census Attributes}
\author{Adult Income ML Study}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage

% --- body ---

\bibliographystyle{plain}
\bibliography{references}

\end{document}
```

## Figure snippet

```latex
\begin{figure}[H]
  \centering
  \includegraphics[width=0.9\linewidth]{figures/fig_01_missingness.png}
  \caption{Missing values by column (EDA-001).}
  \label{fig:missingness}
\end{figure}
```

## Table snippet (booktabs)

```latex
\begin{table}[H]
  \centering
  \caption{Dataset summary}
  \label{tab:dataset-summary}
  \begin{tabular}{ll}
    \toprule
    Metric & Value \\
    \midrule
    Rows & 48,813 \\
    Positive rate & 23.9\% \\
    \bottomrule
  \end{tabular}
\end{table}
```

## Common fixes on Overleaf

- **Unknown image**: upload `figures/` folder; path must be `figures/name.png` not `./figures/`.
- **Unicode**: add `\usepackage[utf8]{inputenc}` (pdfLaTeX) or switch compiler to LuaLaTeX.
- **Underscores in text**: escape as `\_` or wrap in `\texttt{}`.
