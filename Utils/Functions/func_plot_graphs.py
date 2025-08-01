import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import zscore


def plot_graphs(
    df,
    plot_type='scatter',
    x_col=None,
    y_col=None,
    title=None,
    xlabel=None,
    ylabel=None,
    log_scale=False,
    regression=False,
    alpha=0.5,
    figsize=(10, 6),
    line_color='red',
    cmap='viridis',
    annot=True,
    aggfunc='mean',
    show_corr=False,
    remove_outliers=False,
    outlier_method='iqr',
    return_data=False,
    save_path=None
):
    sns.set(style="whitegrid")
    data = df.copy()

    # -------------------------------
    # Outlier Removal
    # -------------------------------
    if remove_outliers and x_col and y_col:
        if outlier_method == 'iqr':
            for col in [x_col, y_col]:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

        elif outlier_method == 'zscore':
            z_scores = np.abs(zscore(data[[x_col, y_col]].dropna()))
            data = data[(z_scores < 3).all(axis=1)]

    # -------------------------------
    # Start Plot
    # -------------------------------
    plt.figure(figsize=figsize)

    if plot_type == 'scatter':
        if regression:
            sns.regplot(
                data=data,
                x=x_col,
                y=y_col,
                scatter_kws={'alpha': alpha},
                line_kws={'color': line_color},
                ci=None
            )
        else:
            sns.scatterplot(
                data=data,
                x=x_col,
                y=y_col,
                alpha=alpha
            )

        if log_scale:
            plt.xscale('log')
            plt.yscale('log')

    elif plot_type == 'boxplot':
        sns.boxplot(data=data, x=x_col, y=y_col)

    elif plot_type == 'bar':
        sns.barplot(data=data, x=x_col, y=y_col, ci=None)

    elif plot_type == 'hist':
        if x_col:
            sns.histplot(data=data, x=x_col, bins=50, kde=True)
        else:
            raise ValueError("x_col must be specified for histogram.")

    elif plot_type == 'heatmap':
        if x_col and y_col:
            pivot_table = data.pivot_table(index=y_col, columns=x_col, aggfunc=aggfunc)
            sns.heatmap(pivot_table, annot=annot, cmap=cmap)
        else:
            raise ValueError("x_col and y_col must be specified for heatmap.")
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type}")

    # -------------------------------
    # Labels & Titles
    # -------------------------------
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel or y_col)
    plt.title(title or f"{plot_type.title()} Plot")

    # -------------------------------
    # Show Correlation on Plot
    # -------------------------------
    if show_corr and x_col and y_col:
        corr = data[[x_col, y_col]].corr().iloc[0, 1]
        plt.text(
            0.05,
            0.95,
            f"Pearson r = {corr:.2f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path)

    plt.show()

    # Optionally return cleaned data
    if return_data:
        return data
