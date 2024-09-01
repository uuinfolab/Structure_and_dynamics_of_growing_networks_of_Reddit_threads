import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

DATA_INPUT_FILE = './fig1b_data.csv'
FIG_OUTPUT_FILE = './fig1b.jpeg'


def main():
    df = pd.read_csv(DATA_INPUT_FILE)

    plt.figure(figsize=(6, 6))
    sns.set_theme(style='ticks', palette='deep')
    g = sns.catplot(
            data=df,
            x='final_judg',
            y='proportion',
            hue='stance',
            kind='bar')

    sns.move_legend(
            g,
            'upper center',
            bbox_to_anchor=[0.5, 0.95],
            ncol=2,
            fontsize=14,
            title=None,
            frameon=False)

    g.ax.tick_params(axis='both', labelsize=12)

    g.ax.set_ylim(0, 1)
    g.ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    g.ax.yaxis.grid(True, 'major', linewidth=1)

    g.ax.set_xlabel('Final judgment', fontsize=14, fontweight='bold')
    g.ax.set_ylabel('Percentage of posts', fontsize=14, fontweight='bold')

    plt.savefig(FIG_OUTPUT_FILE, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
