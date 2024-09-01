import pandas as pd

DATA_INPUT_FILE = '../data-tidy/threads_stats_struct_prop_final.csv'
DATA_OUTPUT_FILE = './fig1b_data.csv'

def main():
    df = pd.read_csv(DATA_INPUT_FILE)
    df = df[["final_judg"]]
    df = df.value_counts(normalize=True).to_frame().reset_index()

    df.set_index('final_judg', inplace=True)
    df = df.rename(index={
        'Not the A-hole': 'NTA',
        'Asshole': 'YTA',
        'No A-holes here': 'NAH',
        'Everyone Sucks': 'ESH'
    })
    df['stance'] = df.index.map(lambda x: 'positive' if x in ['NTA', 'NAH'] else 'negative')


    df.to_csv(
        path_or_buf=DATA_OUTPUT_FILE,
        index=True)

if __name__ == '__main__':
    main()
