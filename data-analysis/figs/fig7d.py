import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

DATA_INPUT_FILE = './fig7_data.csv'
FIG_OUTPUT_FILE = './fig7d.jpeg'

DOT_SIZE = 150
RED_COLOR = sns.color_palette('deep')[3]
GREEN_COLOR = sns.color_palette('deep')[0]
#GREY_COLOR = sns.color_palette('deep')[7]
GREY_COLOR = "dimgray"

def main():
    df = pd.read_csv(DATA_INPUT_FILE)
    df = df[df.subreddit == 'Pinoy']
    df = df.reset_index()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6),)

    # min dot
    ax.scatter(
        x=df["perifery"],
        y=df["cut"],
        s=DOT_SIZE,
        alpha=1,
        color=RED_COLOR,
        label="Min/Max",
        edgecolors="white",
    )

    # max dot
    ax.scatter(
        x=df["star"],
        y=df["cut"],
        s=DOT_SIZE,
        alpha=1,
        color=GREEN_COLOR,
        edgecolors="white",
    )

    ax.hlines(
        xmin=df["perifery"],
        xmax=df["star"],
        y=df["cut"],
        color=GREY_COLOR,
        alpha=0.4,
        lw=4, # line-width
        zorder=0, # make sure line at back
    )

    # iterate through each result and apply the text
    # df should already be sorted
    for i in range(0, df.shape[0]):
        if df["perifery"][i] <= df["star"][i]:
            min_x = df["perifery"][i]
            max_x = df["star"][i]

        else:
            min_x = df["star"][i]
            max_x = df["perifery"][i]

        # min value
        ax.text(
            x=min_x - 0.02,
            y=i + 1 - 0.2,
            s="{:.2f}".format(min_x),
            horizontalalignment="right",
            verticalalignment="top",
            size=10,
            color="black",
            weight="medium",
        )

        # max value
        ax.text(
            x=max_x + 0.02,
            y=i + 1 - 0.2,
            s="{:.2f}".format(max_x),
            horizontalalignment="left",
            verticalalignment="top",
            size=10,
            color="black",
            weight="medium",
        )


        # add thin leading lines towards the y labels
        # to the right of max dot
        ax.plot(
            [max_x + 0.015, 0.4],
            [i + 1, i + 1],
            linewidth=1,
            color="grey",
            alpha=0.4,
            zorder=0,
        )

        # to the left of min dot
        ax.plot(
            [-0.05, min_x - 0.015],
            [i + 1, i + 1],
            linewidth=1,
            color="grey",
            alpha=0.4,
            zorder=0,
        )


        # add ylabels
        label_name = df["cut"][i]
        ax.text(
            x=-0.059,
            y=i + 1,
            s=label_name,
            horizontalalignment="right",
            verticalalignment="center",
            size=14,
            color="black",
            weight="normal",
            )

    ax.set_yticks([])

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xticks([0.0, 0.2, 0.4])
    ax.tick_params(axis="x", pad=20, labelsize=14, labelcolor="black")


    plt.savefig(FIG_OUTPUT_FILE, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
