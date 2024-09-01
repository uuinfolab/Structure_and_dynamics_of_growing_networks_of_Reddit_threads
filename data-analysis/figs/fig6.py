import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

DATA_INPUT_FILE = './fig6_data.csv'
FIG_OUTPUT_FILE = './fig6.jpeg'

RED_COLOR = sns.color_palette('deep')[3]
BLUE_COLOR = sns.color_palette('deep')[0]


def main():
    df = pd.read_csv(DATA_INPUT_FILE)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style='ticks', palette='deep')

    my_palette = {'Vote': RED_COLOR, 'Star': BLUE_COLOR}
    g = sns.catplot(
            data=df,
            x='users',
            y='percentage',
            hue='users',
            kind='box',
            palette=my_palette,
            aspect=1.25
    )

    g.ax.set_alpha(0.6)

    g.ax.set_xlabel('', fontsize=14, fontweight='bold')
    g.ax.set_ylabel('Percentage', fontsize=14, fontweight='bold')

    plt.savefig(FIG_OUTPUT_FILE, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
