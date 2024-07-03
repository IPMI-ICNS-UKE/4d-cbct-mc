from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

from ipmi.common.decorators import convert
from ipmi.fused_types import PathLike

STANDALONE_HEADER = r"""
\documentclass[tikz]{standalone}
\usepackage{float}
\usepackage{tikz}
\usepackage{pdfpages}
\usepackage{pgfplots}
\pgfplotsset{compat=newest}
\usetikzlibrary{external}
\usepgfplotslibrary[groupplots]
"""
BEGIN_DOCUMENT = r"\begin{document}"
END_DOCUMENT = r"\end{document}"


# flake8: noqa: C901
def confusion_matrix_to_tikz(
    cm: np.ndarray,
    cell_labels: np.ndarray | list | None = None,
    class_labels: Sequence[str] | None = None,
    normalize: bool = True,
    normalization_axis: int = 1,
) -> str:
    """Converts a (n, n)-shaped numpy confusion matrix to a pgfplot confusion
    matrix with coloring and saves a tex-file which can directly be compiled by
    use of XeLaTeX.

    Colors are based on the confusion matrix and displayed information in each
    cell corresponds to cell_labels. If no cell_labels are given, cell_labels is set to
    the values of the confusion matrix.
    Note that custom colormap can be specified within the LaTeX code.

    :param cm: (N, N) confusion matrix
    :type cm:
    :param cell_labels: (N, N) cell labels
    :type cell_labels: (M, N, N) or (N, N) provides a label for every cell if list of
    ndarrays or ndarray of 3 dim thefirst axis corresponds to a new line inside the cell
    :param class_labels: N class labels
    :type class_labels:
    :param normalize: whether to normalize the confusion matrix
    :type normalize:
    :param normalization_axis: if normalize is True, the sum along normalization_axis
    is 1
    :type normalization_axis:
    :return: Tikz code of pgf plot
    :rtype:
    """

    if cm.ndim != 2 and cm.shape[0] != cm.shape[1]:
        raise ValueError("Shape mismatch")

    if class_labels and len(class_labels) != cm.shape[0]:
        raise ValueError("Please pass class label for each class")
    if normalize:
        cm = 100 * cm / cm.sum(axis=normalization_axis)[:, None]
    table, nodes = "x y C \n", ""
    size = len(cm)

    if cell_labels is None:
        cell_labels = cm
    if isinstance(cell_labels, list):
        cell_labels = np.array(cell_labels)
    elif cell_labels.ndim == 2:
        cell_labels = cell_labels[None]

    if cm.shape != cell_labels.shape[-2:]:
        raise ValueError(
            "Please provide fitting shape of cell labels and confusion matrix"
        )

    if cell_labels.ndim > 3:
        raise ValueError("cell_labels can hava a maximum of 3 dimensions")

    # handle class labels
    if not class_labels:
        class_labels = (str(label) for label in range(size))

    ticklabels = f"{{{', '.join(class_labels)}}}"

    for (row, col), value in np.ndenumerate(cm):
        table += f"{' ' * 4}{col} {row} {value} \n"
        node_positions = np.linspace(-0.5, 0.5, num=len(cell_labels) + 2)[1:-1]
        for node_position, cell_label in zip(node_positions, cell_labels):
            nodes += (
                rf"{' ' * 4}\node[] at "
                rf"({col},{row+node_position}) {{\tiny {cell_label[row, col]}}} ;"
                "\n"
            )

    body = rf"""
    \begin{{tikzpicture}}
    \begin{{axis}}[
    typeset ticklabels with strut,
    enlargelimits=false,
    tick style={{draw=none}},
    xtick pos=upper,
    xticklabel pos=upper,
    xmin={{{-0.5}}},
    xmax={{{size - 0.5}}},
    ymin={{{-0.5}}},
    ymax={{{size - 0.5}}},
    width={size}cm,
    height={size}cm,
    xlabel=Predicted,
    ylabel=Ground Truth,
    xticklabels={ticklabels},
    xtick={set(range(size))},
    yticklabels={ticklabels},
    ytick={set(range(size))},
    point meta min=0,
    point meta max=100,]
    % colormap name=rocket_r]

    \addplot[
    matrix plot,
    mesh/cols={{{size}}},
    every node near coord/.append style={{yshift=-0.3cm}},
    nodes near coords style={{anchor=center}},
    point meta=explicit,
    ] table [meta=C] {{
    {table.strip()}
    }};
    {{
    {nodes.strip()}
    }}

    \end{{axis}}
    \end{{tikzpicture}}
"""
    custom_packages = r"""
\usepackage[]{mathspec}
\usepgfplotslibrary{colormaps}
\usepackage{pgfplotstable}
\pgfplotsset{compat=newest}
\usepackage{subcaption}
\usepackage{xint}
\usetikzlibrary{calc}
\usetikzlibrary{pgfplots.colormaps}

% Here you can define your custom colormap
% \input{rocket_r.tex}

"""
    filecontent = (
        STANDALONE_HEADER + custom_packages + BEGIN_DOCUMENT + body + END_DOCUMENT
    )

    return filecontent


def cmap_to_tikz(
    cmap: str | mpl.colors.Colormap, steps: int = 100, indent: int = 4
) -> str:
    """Converts a matplotlib Colormap `cmap` to a pgfplot colormap and retuns
    the corresponding code as a string.

    :param cmap:
    :type cmap:
    :param steps:
    :type steps:
    :param indent:
    :type indent:
    :return:
    :rtype:
    """
    if isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(name=cmap)

    tikz_code = f'\\pgfplotsset{{\n{" " * indent}colormap={{{cmap.name}}}{{\n'

    for i in np.linspace(0.0, 1.0, num=steps):
        red, green, blue, alpha = cmap(i)
        tikz_code += f'{" " * 2 * indent}rgb=({red:.6f}, {green:.6f}, {blue:.6f})\n'

    tikz_code += f'{" " * indent}}}\n}}'

    return tikz_code


@convert("output_filepath", converter=Path)
def matplotlib_figure_to_standalone(
    figure: plt.Figure,
    output_filepath: PathLike,
) -> None:
    """Saves `figure` to a .tex file in LaTeX standalone format.

    :param figure:
    :type figure:
    :param output_filepath:
    :type output_filepath:
    :return:
    :rtype:
    """
    # converted to Path
    output_filepath: Path

    body = tikzplotlib.get_tikz_code(figure)
    tikz_code = STANDALONE_HEADER + BEGIN_DOCUMENT + body + END_DOCUMENT

    with open(output_filepath.with_suffix(".tex"), "w") as tikz_file:
        tikz_file.write(tikz_code)
