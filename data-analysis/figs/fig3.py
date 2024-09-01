import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

DATA_INPUT_FILE = './fig3_data.csv'
FIG_OUTPUT_FILE = './fig3.jpeg'

def formatter(x, pos):
    if x == 2.325:
        return '2.325'

    return f'{int(x):,}'


def main():
    df = pd.read_csv(DATA_INPUT_FILE)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style='ticks', palette='deep')
    g = sns.displot(
            data=df,
            x='alpha',
            binwidth=.1,
            kind='hist',
            kde=True,
            height=6,
            aspect=1.6)

    accent1 = sns.color_palette('deep')[3]
    plt.axvline(df.alpha.median(), ls='--', c=accent1)

    g.ax.set_xlim(0, 6)
    g.ax.set_xticks([0, 1, 2, 2.325, 3, 4, 5, 6])
    g.ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(formatter))

    g.ax.tick_params(axis='both', labelsize=12)

    g.ax.set_xlabel('Exponential coefficient ($\\gamma$)', fontsize=14, fontweight='bold')
    g.ax.set_ylabel('Number of threads', fontsize=14, fontweight='bold')

    plt.savefig(FIG_OUTPUT_FILE, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
