import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype

def plot_metrics(metrics, condition_to_latex, title, color_palette):
    # Add a new column to your dataframe with the latex representation
    metrics['condition_latex'] = metrics['condition'].map(condition_to_latex)

    # Order condition_latex as per the condition_to_latex dictionary
    cat_type = CategoricalDtype(categories=condition_to_latex.values(), ordered=True)
    metrics['condition_latex'] = metrics['condition_latex'].astype(cat_type)

    # Compute mean and std for each condition for each metric
    grouped = metrics.groupby('condition_latex')[['mel', 'frechet']].agg(['mean', 'std'])

    fig, axs = plt.subplots(2, 1, figsize=(7, 5.25))

    # Set the main title for the figure
    fig.suptitle(title, fontsize=16)

    # Get color for each bar in the plot
    bar_colors = [color_palette[condition] for condition in grouped.index]

    # Plot mel
    sns.boxplot(x='condition_latex', y='mel', data=metrics, ax=axs[0], palette=color_palette, showfliers=False)
    axs[0].set_ylabel('Mel Spectrogram Loss \u2190')
    axs[0].set_xlabel('') # Remove x-axis label
    axs[0].set_xticklabels(grouped.index, rotation=0, ha='center')

    # Plot frechet
    axs[1].bar(grouped.index, grouped['frechet']['mean'], yerr=grouped['frechet']['std'], color=bar_colors)
    axs[1].set_ylabel('FAD \u2190')
    axs[1].set_xlabel('') # Remove x-axis label
    axs[1].set_xticklabels(grouped.index, rotation=0, ha='center')

    # Adjust the space between plots
    plt.subplots_adjust(hspace=0.1)

    # Remove any unnecessary space around the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Reduce the space between suptitle and the plot
    plt.subplots_adjust(top=0.92)