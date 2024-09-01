import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

DATA_INPUT_FILE = './fig2_data.csv'
FIG_OUTPUT_FILE = './fig2.jpeg'


def main():
    df = pd.read_csv(DATA_INPUT_FILE)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style='ticks', palette='deep')
    g = sns.displot(
            data=df,
            x='post_entropy',
            bins=50,
            kind='hist',
            height=6,
            aspect=1.6)

    accent1 = sns.color_palette('deep')[3]
    plt.axvline(1.3, ls='--', c=accent1)

    accent2 = sns.color_palette('deep')[7]
    plt.axvline(0.65, ls='dotted', c=accent2)
    plt.axvline(1.95, ls='dotted', c=accent2)

    g.ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    g.ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))

    g.ax.tick_params(axis='both', labelsize=12)

    g.ax.set_xlabel('Thread entropy', fontsize=14, fontweight='bold')
    g.ax.set_ylabel('Number of threads', fontsize=14, fontweight='bold')

    plt.savefig(FIG_OUTPUT_FILE, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
