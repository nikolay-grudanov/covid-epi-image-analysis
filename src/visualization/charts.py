"""
Visualization module for medical data.

This module provides functions to create various types of charts and plots
for medical data visualization using Matplotlib and Seaborn.
"""

from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def set_style(style: str = "whitegrid", font_scale: float = 1.2) -> None:
    """
    Set the visualization style.

    Parameters:
    -----------
    style : str, default "whitegrid"
        Seaborn style name
    font_scale : float, default 1.2
        Font scale for text elements
    """
    sns.set_theme(style=style)
    sns.set(font_scale=font_scale)


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (12, 6),
    color: Optional[str] = None,
    rotation: int = 45,
) -> plt.Axes:
    """
    Create a bar chart for categorical data.

    Parameters:
    -----------
    df : pd.DataFrame
        Data to visualize
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis (numeric)
    title : str
        Chart title
    xlabel : str, optional
        X-axis label (defaults to x if not provided)
    ylabel : str, optional
        Y-axis label (defaults to y if not provided)
    figsize : tuple, default (12, 6)
        Figure size
    color : str, optional
        Bar color
    rotation : int, default 45
        Rotation angle for x-axis labels

    Returns:
    --------
    plt.Axes
        Matplotlib axes object

    Example:
    --------
    >>> ax = create_bar_chart(df, "diagnosis", "count", "Распределение диагнозов")
    >>> plt.show()
    """
    set_style()

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(df[x], df[y], color=color)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel(xlabel if xlabel else x, fontsize=12)
    ax.set_ylabel(ylabel if ylabel else y, fontsize=12)

    plt.xticks(rotation=rotation, ha="right")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    return ax


def create_pie_chart(
    df: pd.DataFrame,
    labels: str,
    values: str,
    title: str,
    figsize: tuple = (10, 8),
    autopct: str = "%1.1f%%",
) -> plt.Axes:
    """
    Create a pie chart for categorical distribution.

    Parameters:
    -----------
    df : pd.DataFrame
        Data to visualize
    labels : str
        Column name for labels
    values : str
        Column name for values (numeric)
    title : str
        Chart title
    figsize : tuple, default (10, 8)
        Figure size
    autopct : str, default '%1.1f%%'
        Format for percentage labels

    Returns:
    --------
    plt.Axes
        Matplotlib axes object

    Example:
    --------
    >>> ax = create_pie_chart(df, "finding", "count", "Распределение диагнозов")
    >>> plt.show()
    """
    set_style()

    fig, ax = plt.subplots(figsize=figsize)

    ax.pie(
        df[values],
        labels=df[labels],
        autopct=autopct,
        startangle=90,
        textprops={"fontsize": 10},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.axis("equal")

    plt.tight_layout()

    return ax


def create_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    hue: Optional[str] = None,
    figsize: tuple = (10, 6),
    alpha: float = 0.6,
) -> plt.Axes:
    """
    Create a scatter plot for relationship between two numeric variables.

    Parameters:
    -----------
    df : pd.DataFrame
        Data to visualize
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis
    title : str
        Chart title
    xlabel : str, optional
        X-axis label (defaults to x if not provided)
    ylabel : str, optional
        Y-axis label (defaults to y if not provided)
    hue : str, optional
        Column name for color grouping
    figsize : tuple, default (10, 6)
        Figure size
    alpha : float, default 0.6
        Point transparency

    Returns:
    --------
    plt.Axes
        Matplotlib axes object

    Example:
    --------
    >>> ax = create_scatter_plot(df, "age", "temperature", "Возраст и температура")
    >>> plt.show()
    """
    set_style()

    fig, ax = plt.subplots(figsize=figsize)

    if hue:
        sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=alpha, ax=ax)
    else:
        sns.scatterplot(data=df, x=x, y=y, alpha=alpha, ax=ax)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel(xlabel if xlabel else x, fontsize=12)
    ax.set_ylabel(ylabel if ylabel else y, fontsize=12)

    plt.tight_layout()

    return ax


def create_heatmap(
    df: pd.DataFrame,
    title: str,
    figsize: tuple = (12, 8),
    cmap: str = "YlOrRd",
    annot: bool = True,
    fmt: str = "d",
    linewidths: float = 0.5,
) -> plt.Axes:
    """
    Create a heatmap for correlation or cross-tabulation data.

    Parameters:
    -----------
    df : pd.DataFrame
        Data to visualize (should be a pivot table or correlation matrix)
    title : str
        Chart title
    figsize : tuple, default (12, 8)
        Figure size
    cmap : str, default "YlOrRd"
        Colormap name
    annot : bool, default True
        Whether to annotate cells with values
    fmt : str, default 'd'
        Format string for annotations
    linewidths : float, default 0.5
        Width of grid lines

    Returns:
    --------
    plt.Axes
        Matplotlib axes object

    Example:
    --------
    >>> pivot_df = df.pivot(index="finding", columns="view", values="count")
    >>> ax = create_heatmap(pivot_df, "Диагноз и проекция")
    >>> plt.show()
    """
    set_style()

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(df, annot=annot, fmt=fmt, cmap=cmap, linewidths=linewidths, ax=ax)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    return ax


def create_line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    hue: Optional[str] = None,
    figsize: tuple = (12, 6),
    marker: str = "o",
) -> plt.Axes:
    """
    Create a line plot for time series or trend data.

    Parameters:
    -----------
    df : pd.DataFrame
        Data to visualize
    x : str
        Column name for x-axis (usually temporal)
    y : str
        Column name for y-axis
    title : str
        Chart title
    xlabel : str, optional
        X-axis label (defaults to x if not provided)
    ylabel : str, optional
        Y-axis label (defaults to y if not provided)
    hue : str, optional
        Column name for color grouping
    figsize : tuple, default (12, 6)
        Figure size
    marker : str, default 'o'
        Marker style

    Returns:
    --------
    plt.Axes
        Matplotlib axes object

    Example:
    --------
    >>> ax = create_line_plot(df, "date", "count", "Временные тренды")
    >>> plt.show()
    """
    set_style()

    fig, ax = plt.subplots(figsize=figsize)

    if hue:
        sns.lineplot(data=df, x=x, y=y, hue=hue, marker=marker, ax=ax)
    else:
        sns.lineplot(data=df, x=x, y=y, marker=marker, ax=ax)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel(xlabel if xlabel else x, fontsize=12)
    ax.set_ylabel(ylabel if ylabel else y, fontsize=12)

    plt.tight_layout()

    return ax


def save_chart(ax: plt.Axes, filepath: str, dpi: int = 300) -> None:
    """
    Save a chart to file.

    Parameters:
    -----------
    ax : plt.Axes
        Matplotlib axes object
    filepath : str
        Path to save the chart
    dpi : int, default 300
        Resolution in dots per inch

    Example:
    --------
    >>> ax = create_bar_chart(df, "diagnosis", "count", "Title")
    >>> save_chart(ax, "data/results/visualizations/chart.png")
    """
    fig = ax.get_figure()
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to: {filepath}")
