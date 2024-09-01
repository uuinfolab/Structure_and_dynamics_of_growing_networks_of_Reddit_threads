import os
import pandas as pd
import numpy as np

DATA_INPUT_DIR = '../data-analysis/network_data/networks_in_time'
DATA_INPUT_FILE_PREFIX = 'user_edgelist_in_time'
DATA_OUTPUT_FILE = './fig5_data.csv'

def main():
    tmp_dict = {}
    res_df = pd.DataFrame()

    files = [f for f in os.listdir(DATA_INPUT_DIR) if \
            (f.endswith('.csv') and f.startswith(DATA_INPUT_FILE_PREFIX))]
    vars = ['density', 'diameter', 'clust_coeff', 'aspl']

    for idx, var in enumerate(vars):
        for file in files:
            file_path = os.path.join(DATA_INPUT_DIR, file)
            df = pd.read_csv(file_path, low_memory=False, index_col='timestamp')

            try:
                tmp_dict.update({file: df[vars[idx]]})
            except KeyError:
                continue

        df = pd.DataFrame(tmp_dict)

        if idx == 2:
            df.replace(0.0, np.nan, inplace=True) # remove zeros at the beginning of cust coeff

        df.reset_index(drop=True, inplace=True)
        df = df.ffill()
        df['average'] = df.mean(axis=1)
        df['interv'] = df.std(axis=1)
        df['measure'] = var
        df['t'] = range(len(df['average']))
        res_df = pd.concat([res_df, df[['measure', 't', 'average', 'interv']]], ignore_index=True)

    res_df = res_df.rename({
        'measure': 'measure',
        't': 't',
        'average': 'mean',
        'interv': 'std'}, axis='columns')

    res_df.to_csv(
        path_or_buf=DATA_OUTPUT_FILE,
        index=True)

if __name__ == '__main__':
    main()
