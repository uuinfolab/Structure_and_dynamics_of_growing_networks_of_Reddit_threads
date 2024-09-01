import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

DATA_INPUT_FILE = './fig5_data.csv'
FIG_OUTPUT_FILE = './fig5d.jpeg'

def main():
    df = pd.read_csv(DATA_INPUT_FILE)
    df = df[df.measure == 'diameter']

    plt.figure(figsize=(6, 6))
    sns.set_theme(style='ticks', palette='deep')

    g = sns.relplot(
            data=df,
            x='t',
            y='mean',
            #err_kws=err,
            kind='line')

    g.ax.set_xscale('log')
    plt.fill_between(
        x=list(df.t),
        y1=list(df['mean'] - df['std']),
        y2=list(df['mean'] + df['std']),
        alpha=.1
        )

    g.ax.tick_params(axis='both', labelsize=12)
    g.ax.xaxis.grid(True, 'major', linewidth=.5, color='dimgray')
    g.ax.xaxis.grid(True, "minor", linewidth=.25)
    g.ax.yaxis.grid(True, 'major', linewidth=.5, color='dimgray')

    g.ax.set_xlabel('Growing number of edges in time (log)', fontsize=14, fontweight='bold')
    g.ax.set_ylabel('Diameter', fontsize=14, fontweight='bold')

    plt.savefig(FIG_OUTPUT_FILE, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
