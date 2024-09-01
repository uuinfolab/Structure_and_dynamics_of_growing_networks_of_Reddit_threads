import pandas as pd

DATA_INPUT_FILE = '../data-tidy/threads_stats_struct_prop_final.csv'
DATA_OUTPUT_FILE = './fig6_data.csv'

def main():
    df = pd.read_csv(DATA_INPUT_FILE)
    df['voters_perc'] = 100 - df.didnt_vote_perc
    df = df[['voters_perc', 'first_level_users_perc']]
    df.index.name = 'thread'

    df = df.rename({
        'voters_perc': 'Vote',
        'first_level_users_perc': 'Star'
    }, axis='columns')
    df['id'] = df.index.values

    df = pd.melt(df,
            id_vars=['id'],
            value_vars=['Vote', 'Star'],
            var_name='users',
            value_name='percentage')
    df = df[['users', 'percentage']]

    df.to_csv(
        path_or_buf=DATA_OUTPUT_FILE,
        index=True)

if __name__ == '__main__':
    main()
