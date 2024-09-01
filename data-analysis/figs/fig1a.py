import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

DATA_INPUT_FILE = './fig1a_data.csv'
FIG_OUTPUT_FILE = './fig1a.jpeg'


def main():
    df = pd.read_csv(DATA_INPUT_FILE)

    plt.figure(figsize=(6, 6))
    sns.set_theme(style='ticks', palette='deep')
    g = sns.displot(
            data=df,
            x='num_comments',
            kind='ecdf')

    g.ax.set_xlim(0, 15000)
    g.ax.xaxis.set_major_locator(ticker.MultipleLocator(2000))
    g.ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
    g.ax.tick_params(axis='x', labelsize=11)
    g.ax.tick_params(axis='y', labelsize=12)

    g.ax.set_ylim(0.7, 1.01)
    g.ax.set_yticks([0.75, 0.8, 0.85, 0.9, 0.95, 1])

    g.ax.xaxis.grid(True, 'major', linewidth=1)
    g.ax.yaxis.grid(True, 'major', linewidth=1)

    g.ax.set_xlabel('Comments per thread', fontsize=14, fontweight='bold')
    g.ax.set_ylabel('ECDF', fontsize=14, fontweight='bold')

    plt.savefig(FIG_OUTPUT_FILE, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
