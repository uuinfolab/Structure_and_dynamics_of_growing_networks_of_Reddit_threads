import pandas as pd

DATA_INPUT_FILE = '../data-tidy/avg_speeds.csv'
DATA_OUTPUT_FILE = './fig7_data.csv'

def main():
    df = pd.read_csv(DATA_INPUT_FILE)
    df = df[df['time_interval'] == "1 minute"]
    df = df[["subreddit", "cut_bin", "avg_speed_nodes_star", "avg_speed_node_per"]]

    df = df.rename({
        'subreddit': 'subreddit',
        'cut_bin': 'cut',
        'avg_speed_node_per': 'perifery',
        'avg_speed_nodes_star': 'star'}, axis='columns')

    df.to_csv(
        path_or_buf=DATA_OUTPUT_FILE,
        index=True)

if __name__ == '__main__':

    main()
