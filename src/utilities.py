import numpy as np
import pandas as pd
from scipy.stats import entropy, ttest_ind, ttest_rel
from datetime import timedelta
from collections import Counter
import seaborn as sns
import igraph as ig
from tqdm import tqdm
import os

''' FUNCTIONS TO PROCESS DATA '''

def in_18_hours_func(df):

    df['in_18_hours'] = False # default:illegal
    submission_data = df[df['type'] == 'sub'][['id', 'created']].set_index('id') # save date for each submission to retrieve it optimally
    df['submission_created'] = df['submission_id'].map(submission_data.to_dict()['created'])
    df['legal_time'] = df['submission_created'] + pd.Timedelta(18, 'h', hours=18)
    df['in_18_hours'] = df['created'] < df['legal_time'] # legal if comment within 18 h

def legal_func(df):
    df['voter'] = 1 # default: voter
    df['legal'] = 0 # default: illegal

    for index, row in df.iterrows():

        ### if is not a comment OR did not vote
        if row.text_flair_list=='[]':
            df.at[index, 'voter'] = 0 # not a voter
            df.at[index, 'legal'] = np.nan # not applicable

        ### if 1st level AND within 18 hours AND only 1 vote
        ### (ALL conditions True necessary to make a vote legal)
        elif row.depth==0 and row.in_18_hours==1 and row.number_text_flairs==1: 
            df.at[index, 'legal'] = 1

        # default case: illegal
        else:
            continue

def vocal_func(df):
    df['vocal'] = 1 # default: vocal

    for index, row in df.iterrows():


        if row.voter != 1:
            df.at[index, 'vocal'] = np.nan # not applicable

        ### if no verbs then it's silent
        elif row.verbs == 0:
            df.at[index, 'vocal'] = 0

        # default case
        else:
            continue

''' COMPUTE DISAGREEMENT '''

def compute_post_entropy(df):
    df = df[df.voter==1]
    unique_votes, counts = np.unique(list(df.text_flair), return_counts=True)
    prob = counts / len(df.text_flair)
    entr = round(entropy(prob, base=2), 2)
    return entr

def compute_entropy_in_time(sub_df, final_judg):
    sub_df.sort_values(by='created', inplace=True)
    #df = sub_df
    df_star = sub_df[sub_df.depth<1]
    df_per = sub_df[sub_df.depth>=1]
    df_star['subgraph'] = 'star'
    df_per['subgraph'] = 'per'
    df_star['entropy_in_time'] = df_star.apply(lambda x: compute_post_entropy(df_star[df_star.created <= x.created]), axis=1)
    df_per['entropy_in_time'] = df_per.apply(lambda x: compute_post_entropy(df_per[df_per.created <= x.created]), axis=1)
    df_star['final_judg'] = final_judg
    df_per['final_judg'] = final_judg
    df=pd.concat([df_star, df_per])
    return df[['submission_id', 'final_judg', 'created', 'entropy_in_time', 'subgraph']]

def perc(v, tot):
    ''' compute percentage of a variable '''
    try:
        return round((v/tot)*100, 2)
    except ZeroDivisionError:
        return 0

''' STATISTICAL TESTS '''

def ttests(vars, static_networks_df):

    alpha = 0.005
    for var in vars:
        t_stat, p_value = ttest_ind(static_networks_df.post_entropy.values.reshape(-1, 1),
                                          static_networks_df[var].values.reshape(-1, 1), equal_var=False)
        print('Ind', var, p_value<alpha, p_value)

        t_stat, p_value = ttest_rel(static_networks_df.post_entropy.values.reshape(-1, 1),
                                          static_networks_df[var].values.reshape(-1, 1))
        print('Dep', var, p_value<alpha, p_value)

def scale_free_test():
    p = './data-analysis/network-data/networks_in_time/'
    file_list = [f for f in os.listdir(p) if (f.endswith('.csv') and f.startswith('user_edgelist_in_time'))]
    pbar = tqdm(total=6366)
    dict_= {}
    for file_name in file_list:
        file_path = os.path.join(p, file_name)
        df = pd.read_csv(file_path, low_memory=False)
        df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S.%f')

        G = ig.Graph.TupleList(df[['from', 'to']].itertuples(index=False), directed=True)
        degree_distribution = G.degree() # Get the degree distribution
        fit = ig.power_law_fit(degree_distribution) # Fit a power-law distribution
        is_scale_free = fit.alpha > 2 # Check if the graph follows a scale-free distribution
        dict_.update({file_name: [is_scale_free, fit.alpha, fit.D, fit.p]})
        pbar.update(1)
    df = pd.DataFrame(dict_)
    df.to_csv('./data-tidy/scale_free_test.csv', index=False)

def scale_free_analysis():
    sns.set_theme()
    print('Executing function: scale_free_analysis')
    df = pd.read_csv('./data-tidy/scale_free_test.csv', low_memory=False)
    df=df.T
    df.columns = ['is_scale_free', 'alpha', 'D', 'p_value']
    df['alpha'] = pd.to_numeric(df['alpha'], errors='coerce')
    df['D'] = pd.to_numeric(df['D'], errors='coerce')
    df['p_value'] = pd.to_numeric(df['p_value'], errors='coerce')

    print('Scale free:', len(df[df['is_scale_free'] == 'True']), '\n Non scale free:',
          len(df[df['is_scale_free'] == 'False']))

    print(df.describe(percentiles=[.12]))
    


''' FUNCTIONS TO WORK WITH TIME '''
def minute_rounder(t):
    '''
    Rounds to nearest minute by adding a timedelta, minute if second >= 30.

    Notes: used in 'plot_subgraphs_growth' function contained in 'UU_functions' file.

    :param t: datetime object
    :return: datetime object rounded to nearest minute
    '''

    return (t.map(lambda x : x.replace(second=0, microsecond=0, minute=x.minute, hour=x.hour)
               +timedelta(minutes=x.second//30)))

def ten_minute_rounder(dt):
    return (dt.map(lambda x : x.replace(second=0, microsecond=0, minute=x.minute//10*10, hour=x.hour)))#

def hour_rounder(t):
    return (t.map(lambda x : x.replace(second=0, microsecond=0, minute=0, hour=x.hour)
               +timedelta(hours=x.minute//30)))

def z_normalize(df, col):
    return (df[col] - df[col].mean()) / df[col].std(ddof=0)  # z normalization

''' RECIPROCITY '''

def compute_groups(lst):
    groups = {}
    for tup in lst:
        x, y, _ = tup
        if (x, y) in groups:
            groups[(x, y)].append(tup)
        elif (y, x) in groups:
            groups[(y, x)].append(tup)
        else:
            groups[(x, y)] = [tup]
    grouped_tuples = list(groups.values())
    return grouped_tuples

def compute_reciprocity(lst):
    grouped_tuples = compute_groups(lst)
    yes_rec = [group for group in grouped_tuples if len(group) > 1]
    no_rec = [group for group in grouped_tuples if len(group) == 1]

    yes_rec_actors = [group[0][0] for group in yes_rec] + [group[0][1] for group in yes_rec]
    yes_rec_actors = Counter(yes_rec_actors)  # dict{actor: how many reciprocical edges has}

    no_rec_actors = [group[0][0] for group in no_rec] + [group[0][1] for group in no_rec]

    return yes_rec_actors