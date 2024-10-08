import pandas as pd

DATA_INPUT_FILE = '../data-tidy/threads_stats_struct_prop_final.csv'
DATA_OUTPUT_FILE = './fig2_data.csv'

def main():
    df = pd.read_csv(DATA_INPUT_FILE)
    df = df[["post_entropy"]]
    df.index.name = 'thread'

    df.to_csv(
        path_or_buf=DATA_OUTPUT_FILE,
        index=True)

if __name__ == '__main__':
    main()
