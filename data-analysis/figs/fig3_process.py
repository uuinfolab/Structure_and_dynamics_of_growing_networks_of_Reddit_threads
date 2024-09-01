import pandas as pd

DATA_INPUT_FILE = '../data-tidy/scale_free_test.csv'
DATA_OUTPUT_FILE = './fig3_data.csv'

def main():
    df = pd.read_csv(DATA_INPUT_FILE, low_memory=False)
    df = df.T
    df.columns = ['is_scale_free', 'alpha']
    df = df[["alpha"]]
    df.index.name = 'network'

    df.to_csv(
        path_or_buf=DATA_OUTPUT_FILE,
        index=True)

if __name__ == '__main__':
    main()
