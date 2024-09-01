import itertools
from IPython.display import display
import igraph as ig
from igraph import Graph
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
from tqdm import tqdm
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import networkx as nx
from pylab import rcParams
rcParams['figure.figsize'] = 20, 15
from datetime import timedelta
from utilities import *



''' STRUCTURAL PROPERTIES '''

def compute_user_network(draw=True):
    node_attr_df = pd.read_csv("../data-analysis/network_data/user_networks_nodelist/user_nodelist.csv", low_memory=False)
    df = pd.read_csv("../data-tidy/user_edgelist.csv", low_memory=False)
    df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S.%f')
    merged_df = pd.DataFrame()
    for subm_id, sub_df in df.groupby('root'):
        pbar2 = tqdm(total=len(sub_df))
        sub_df.sort_values(by='created', inplace=True)
        sub_df.reset_index(drop=True, inplace=True)
        sub_df['timestamp'] = sub_df.index

        # DRAW
        if draw:
            G = nx.from_pandas_edgelist(
                sub_df, source='from', target='to',
                edge_attr=["created", "timestamp"],
                create_using=nx.MultiDiGraph(),
            )
            nx.set_node_attributes(G, node_attr_df.set_index('author').to_dict(
                'index'))  # in the user nodelist file, for each post id, authors must be unique
            # print(G.nodes(data=True))
            # print(G.edges(data=True))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=False, node_size=8, font_size=6, font_color='grey',
                    node_color='skyblue', edge_color='grey', linewidths=0.1)
            plt.show()


        # NETWORK IN TIME
        density_list = []
        diameter_list = []
        clust_coeff_list = []
        aspl_list = []
        number_of_nodes = []
        subgraph = nx.DiGraph()  # simple and undirected
        for t in range(len(sub_df['timestamp'])):
            # selected_edges = [(u, v, data) for u, v, data in G.edges(data=True) if data.get('timestamp') <= t]
            subgraph.add_edge(sub_df['from'][t], sub_df['to'][t], timestamp=t, created=sub_df['created'][t])
            # nx.set_node_attributes(subgraph, node_attr_df[node_attr_df.author.isin(subgraph.nodes())].set_index('author').to_dict('index'))
            # CHECK 1
            # print(subgraph.edges(data=True))
            # print(t, subgraph.nodes(data=True))
            # CHECK 2
            # nx.draw(subgraph)
            # plt.show()

            subgraph2 = ig.Graph.from_networkx(subgraph)
            # degree = dict(subgraph.degree()) 
            if t==0:
                clust_coeff_list.append(0)
                diameter_list.append(0)
                number_of_nodes.append(2)
                aspl_list.append(0)
            else:
                number_of_nodes.append(subgraph2.vcount()) #number_of_nodes.append(len(subgraph.nodes()))
                # aspl_list.append(nx.average_shortest_path_length(subgraph))
                aspl_list.append(np.mean(subgraph2.average_path_length(directed=False)))
                diameter_list.append(subgraph2.diameter()) # in nx only possible for undirected graphs
                if number_of_nodes[t] > number_of_nodes[t-1]: # if new node is added, clust coeff is not updated
                    clust_coeff_list.append(clust_coeff_list[t-1])
                else:
                    clust_coeff_list.append(subgraph2.transitivity_undirected())#(nx.average_clustering(subgraph))  # (nx.Graph(subgraph))  # not implemented for multigraph type
            density_list.append(subgraph2.density())
            pbar2.update(1)

        sub_df['density'] = density_list
        sub_df['diameter'] = diameter_list
        sub_df['clust_coeff'] = clust_coeff_list
        sub_df['aspl'] = aspl_list
        #print(sub_df.info())
        #merged_df = pd.concat([merged_df, sub_df])
        sub_df.to_csv(f'../data-tidy/networks_in_time_2/user_edgelist_in_time_{str(subm_id)}.csv', index=False)
    #merged_df.to_csv('../data-tidy/user_edgelist_in_time.csv', index=False)

def include_struct_prop():
    # import thread stats file
    thread_stats = pd.read_csv('../data-tidy/threads_stats.csv')
    p = '../data-tidy/networks_in_time/'
    file_list = [f for f in os.listdir(p) if (f.endswith('.csv') and f.startswith('user_edgelist_in_time'))]
    density_dict = {}
    diameter_dict = {}
    clust_coeff_dict = {}
    aspl_dict = {}
    for f in file_list:
        df = pd.read_csv(p+f, index_col=0)
        dictionary = df.tail(1).to_dict()
        density_dict.update(dictionary['density'])
        diameter_dict.update(dictionary['diameter'])
        clust_coeff_dict.update(dictionary['clust_coeff'])
        aspl_dict.update(dictionary['aspl'])
    thread_stats['density'] = thread_stats['subm_id'].map(density_dict)
    thread_stats['diameter'] = thread_stats['subm_id'].map(diameter_dict)
    thread_stats['clust_coeff'] = thread_stats['subm_id'].map(clust_coeff_dict)
    thread_stats['aspl'] = thread_stats['subm_id'].map(aspl_dict)
    thread_stats.to_csv("../data-tidy/threads_stats_struct_prop.csv", index=True)


''' REIPROCITY'''
def reciprocity_in_time():
    print('Executing function: reciprocity_in_time')
    p = './data-analysis/network-data/networks_in_time/'
    file_list = [f for f in os.listdir(p) if (f.endswith('.csv') and f.startswith('user_edgelist_in_time'))]
    ratio_recipr_edges_list = []
    ratio_bursts_edges_list = []
    pbar = tqdm(total=6366)
    i=0
    for file_name in file_list:
        i+=1
        file_path = os.path.join(p, file_name)
        df = pd.read_csv(file_path, low_memory=False)
        if df.empty:
            continue
        df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S.%f')
        df.dropna(subset=['from', 'to'], inplace=True) # drop entire row if from or to have a nan
        if df.empty:
            continue

        df['from'] = df['from'].astype(str)
        df['to'] = df['to'].astype(str)

        # auxiliary column (optimized version)
        #df['fromto'] = [a+b for a, b in zip(df['from'], df['to'])]
        #df['fromto_alphabetically'] = [''.join(sorted([a,b])) for a, b in zip(df['from'], df['to'])]
        df['fromto'] = df['from'] + df['to']

        try:
            df['fromto_alphabetically'] = df[['from', 'to']].apply(lambda x: ''.join(sorted(x)), axis=1)
        except ValueError:
            print(df[['from', 'to']])
            exit(0)
        df['alternate'] = df.groupby('fromto_alphabetically')['fromto'].apply(lambda x: (x == x.shift(1)).astype(int))
        df['bursts'] = df['alternate'].cumsum() #df['bursts'] = df.groupby(['from', 'to'])['from'].transform('cumcount') # NO

        edgelist=[(b, a) for a, b in zip(df['from'], df['to'])]
        df['reciprocity_corrected'] = [1 if list(zip(df['from'], df['to'])).count(e)>1 else 0 for e in edgelist]
        df['reciprocity_corrected'] = df['reciprocity_corrected']-df['alternate']
        df['reciprocity_corrected'] = df['reciprocity_corrected'].apply(lambda x: 0 if x==-1 else x) # replace -1 with 0
        df['reciprocity_increm'] = df['reciprocity_corrected'].cumsum()

        #df['ratio_recipr_edges'] = df['reciprocity_increm'] / df['timestamp']
        #df['ratio_bursts_edges'] = df['bursts'] / df['timestamp']

        # matrices
        #ratio_bursts_edges_list.append(df['ratio_bursts_edges'])
        #ratio_recipr_edges_list.append(df['ratio_recipr_edges'])

        df.to_csv(f'../data-tidy/recipr_in_time/{file_name}.csv', index=False)
        pbar.update(1)


def reciprocity_analysis():
    print('Executing function: reciprocity_analysis')
    p = '../data-tidy/recipr_in_time/'
    file_list = [f for f in os.listdir(p) if f.endswith('.csv')]
    df_list = [pd.read_csv(p+file_name, low_memory=False) for file_name in file_list]

    ops = [df['to'].iloc[0] for df in df_list if not df.empty]
    stars = [df[df['to'].isin(ops)] for df in df_list if not df.empty]
    perifs = [df[~df['to'].isin(ops)] for df in df_list if not df.empty]

    df_list = stars
    recipr_list_star = [list(df.reciprocity_increm/df.timestamp) for df in df_list if not df.empty]
    #bursts_list = [list(df.bursts/df.timestamp) for df in df_list if not df.empty]
    recipr_list_star = [np.array(tpl) for tpl in zip(*itertools.zip_longest(*recipr_list_star))] # makes the inner lists of equal length filling the gaps with None
    #bursts_list = [np.array(tpl) for tpl in zip(*itertools.zip_longest(*bursts_list))]
    recipr_list_star = np.array(recipr_list_star, dtype=np.float64)
    #bursts_list = np.array(bursts_list, dtype=np.float64)

    df_list = perifs
    recipr_list_perif = [list(df.reciprocity_increm/df.timestamp) for df in df_list if not df.empty]
    recipr_list_perif = [np.array(tpl) for tpl in zip(*itertools.zip_longest(*recipr_list_perif))]
    recipr_list_perif = np.array(recipr_list_perif, dtype=np.float64)

    plt.plot(np.nanmean(recipr_list_star, axis=0), label='star')  # vertical mean
    plt.plot(np.nanmean(recipr_list_perif, axis=0), label='periphery')  # vertical mean
    #plt.plot(np.nanmean(bursts_list, axis=0), label='reciprocity')
    plt.xlabel('Growing number of edges in time', fontsize=20)
    plt.ylabel('Ratio with respect to number of edges', fontsize=20)
    plt.grid()
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=20)
    plt.show()


def include_recipr():
    # import thread stats file
    thread_stats = pd.read_csv('../data-tidy/threads_stats_struct_prop.csv')
    p = '../data-tidy/recipr_in_time/'
    file_list = [f for f in os.listdir(p) if (f.endswith('.csv') and f.startswith('user_edgelist_in_time'))]
    recipr_dict = {}
    bursts_dict = {}
    recipr_ratio_dict = {}
    bursts_ratio_dict = {}
    for f in file_list:
        df = pd.read_csv(p+f, index_col=0)
        df['recipr/edges'] = df['reciprocity_increm'] / df['timestamp']
        df['bursts/edges'] = df['bursts'] / df['timestamp']
        dictionary = df.tail(1).to_dict()
        recipr_dict.update(dictionary['reciprocity_increm'])
        bursts_dict.update(dictionary['bursts'])
        recipr_ratio_dict.update(dictionary['recipr/edges'])
        bursts_ratio_dict.update(dictionary['bursts/edges'])
    thread_stats['reciprocity'] = thread_stats['subm_id'].map(recipr_dict)
    thread_stats['bursts'] = thread_stats['subm_id'].map(bursts_dict)
    thread_stats['reciprocity_ratio'] = thread_stats['subm_id'].map(recipr_ratio_dict)
    thread_stats['bursts_ratio'] = thread_stats['subm_id'].map(bursts_ratio_dict)
    thread_stats.to_csv("../data-tidy/threads_stats_struct_prop_and_rec.csv", index=False)

''' NETWORK GROWTH AND SPEED '''

def subgraphs_growth(subreddit, time_interval, edges, cut1, cut2):
    cut1 = cut1 / 60 if time_interval=='1h' else cut1
    cut2 = cut2 / 60 if time_interval=='1h' else cut2
    if edges:
        path = f'./data-tidy/growth_in_time_edges_{time_interval}/' if subreddit == 'aita' else f'../data-tidy/growth_{subreddit}_edges_{time_interval}/'
        var = 'edges'
    else:
        path = f'./data-tidy/growth_in_time_nodes_{time_interval}/' if subreddit == 'aita' else f'../data-tidy/growth_{subreddit}_nodes_{time_interval}/'
        var = 'nodes'
    file_list = [f for f in os.listdir(path) if (f.endswith('.csv'))]
    star_df_list, per_df_list, star_velocity, per_velocity = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    star_freq_df, per_freq_df = pd.DataFrame(), pd.DataFrame()
    i=0
    for f in file_list:
        i+=1
        if f.startswith('star'):
            star_df = pd.read_csv(path + f, index_col=0)
            star_df.velocity.fillna(0, inplace=True) # new
            star_df['freq'] = star_df.delta_t / star_df.number # new
            star_df_list = star_df_list.merge(star_df[['number']], how='outer', left_index=True, right_index=True, suffixes=['', f'{str(i + 1)}'])
            if ((star_df.index.max() - star_df.index.min()) > cut1) & ((star_df.index.max() - star_df.index.min()) <= cut2):
                star_velocity = star_velocity.merge(star_df[['velocity']], how='outer', left_index=True, right_index=True, suffixes=['', f'{str(i + 1)}'])
                star_freq_df = star_freq_df.merge(star_df[['freq']], how='outer', left_index=True, right_index=True, suffixes=['', f'{str(i + 1)}'])

        else:
            per_df = pd.read_csv(path + f, index_col=0)
            per_df.velocity.fillna(0, inplace=True) # new
            per_df['freq'] = per_df.delta_t / per_df.number # new
            per_df_list = per_df_list.merge(per_df[['number']], how='outer', left_index=True, right_index=True, suffixes=['', f'{str(i + 1)}'])
            if ((per_df.index.max() - per_df.index.min()) > cut1) & ((per_df.index.max() - per_df.index.min()) <= cut2):
                per_velocity = per_velocity.merge(per_df[['velocity']], how='outer', left_index=True, right_index=True, suffixes=['', f'{str(i + 1)}'])
                per_freq_df = per_freq_df.merge(per_df[['freq']], how='outer', left_index=True, right_index=True, suffixes=['', f'{str(i + 1)}'])

        # prolonging the constant number of nodes reached
        star_df_list.fillna(method='ffill', inplace=True)
        per_df_list.fillna(method='ffill', inplace=True)

        # averaging number of nodes across time intervals
        star_df_list['avg'] = star_df_list.mean(axis=1)
        per_df_list['avg'] = per_df_list.mean(axis=1)

        star_df_list = star_df_list[['avg']]
        per_df_list = per_df_list[['avg']]

    max_star, min_star, max_per, min_per = np.nanpercentile(star_velocity, 95), np.nanpercentile(star_velocity,5), np.nanpercentile(per_velocity, 95), np.nanpercentile(per_velocity, 5)
    star_velocity['avg_velocity'] = star_velocity.mean(axis=1, skipna=True)
    per_velocity['avg_velocity'] = per_velocity.mean(axis=1, skipna=True)
    star_freq_df['avg_freq'] = star_freq_df.mean(axis=1, skipna=True)
    per_freq_df['avg_freq'] = per_freq_df.mean(axis=1, skipna=True)
    #print(star_freq_df, per_freq_df)
    print(f'{subreddit, var, time_interval}')
    print('AVG SPEED star: ', round(star_velocity['avg_velocity'].mean(), 2))
    print('AVG SPEED per: ', round(per_velocity['avg_velocity'].mean(), 2))
    print('95% SPEED star: ', round(max_star, 2), '\n5% SPEED star: ', round(min_star, 2), '\n95% SPEED per: ', round(max_per, 2), '\n5% SPEED per: ', round(min_per, 2))

    #return star_df_list, per_df_list


def subgraphs_growth_pre(subreddit, path_to_subreddit, time_interval, edges=True):

    path = '../data-tidy/networks_in_time/' if subreddit == 'aita' else path_to_subreddit
    file_list = [f for f in os.listdir(path) if (f.endswith('.csv') and f.startswith('user_edgelist_in_time'))]
    avg_len_star = []
    avg_len_per = []
    for f in file_list:
        df = pd.read_csv(path + f)
        try:
            df['created'] = pd.to_datetime(df['created'])
        except KeyError:
            df['timestamp'] = pd.to_datetime(df['timestamp']) # for ukraine and jokes dataset
        try:
            op=df['to'][0]
            if subreddit == 'war' or subreddit == 'geopolitics':
                op = df['from'][0]
                # keep first row but filter all other rows if to column is NaN
                df2 = df[df['to'].notna()]
                df2 = df2.append(df.iloc[0])
                df2.sort_values(by='timestamp', inplace=True)
                #print(df, '\nPost\n', df2, '\n\n')
                df = df2
        except KeyError:
            continue
        star_df = df[(df['to']==op)]
        per_df = df[(df['to']!=op)]

        # EDGES
        star_df.sort_values(by='timestamp', inplace=True)
        star_df.reset_index(inplace=True, drop=True)
        per_df.sort_values(by='timestamp', inplace=True)
        per_df.reset_index(inplace=True, drop=True)

        # NODES
        star_df['nodes'] = np.nan
        per_df['nodes'] = np.nan
        authors = [op]
        s=1
        p=0
        for idx, row in star_df.iterrows():
            if row['from'] not in authors:
                star_df.at[idx, 'nodes'] = s
                s+=1
            else:
                try:
                    star_df.at[idx, 'nodes'] = star_df.at[idx-1, 'nodes']
                except KeyError:
                    star_df.at[idx, 'nodes'] = 0
            authors.append(row['from'])
        for idx, row in per_df.iterrows():
            if row['from'] not in authors:
                per_df.at[idx, 'nodes'] = p
                p+=1
            else:
                try:
                    per_df.at[idx, 'nodes'] = per_df.at[idx-1, 'nodes']
                except KeyError:
                    per_df.at[idx, 'nodes'] = 0
            authors.append(row['from'])

        if edges:
            star_df['number'] = star_df.index
            per_df['number'] = per_df.index
            var = 'edges'
        else:
            star_df['number'] = star_df['nodes']
            per_df['number'] = per_df['nodes']
            var = 'nodes'

        # TIME INTERVALS
        try:
            star_df['time_rounded'] = hour_rounder(star_df.created) if time_interval == '1h' else ten_minute_rounder(star_df.created) if time_interval == '10m' else minute_rounder(star_df.created)
            per_df['time_rounded'] = hour_rounder(per_df.created) if time_interval == '1h' else ten_minute_rounder(per_df.created) if time_interval == '10m' else minute_rounder(per_df.created)
        except AttributeError: # for ukraine and jokes dataset
            star_df['time_rounded'] = hour_rounder(star_df.timestamp) if time_interval == '1h' else ten_minute_rounder(star_df.timestamp) if time_interval == '10m' else minute_rounder(star_df.timestamp)
            per_df['time_rounded'] = hour_rounder(per_df.timestamp) if time_interval == '1h' else ten_minute_rounder(per_df.timestamp) if time_interval == '10m' else minute_rounder(per_df.timestamp)
        approx_factor = 3600 if time_interval == '1h' else 600 if time_interval == '10m' else 60
        star_df['time_int'] = star_df['time_rounded'].diff().fillna(timedelta(0)).apply(
            lambda x: x.total_seconds() / approx_factor)
        per_df['time_int'] = per_df['time_rounded'].diff().fillna(timedelta(0)).apply(
            lambda x: x.total_seconds() / approx_factor)
        star_df['time_int'] = star_df['time_int'].cumsum()
        per_df['time_int'] = per_df['time_int'].cumsum()
        rounded_star_df = star_df.groupby(['time_rounded'])['number', 'time_int'].max()
        rounded_per_df = per_df.groupby(['time_rounded'])['number', 'time_int'].max()
        rounded_star_df['time_int'] = rounded_star_df['time_int']*10 if time_interval == '10m' else rounded_star_df['time_int']
        rounded_per_df['time_int'] = rounded_per_df['time_int']*10 if time_interval == '10m' else rounded_per_df['time_int']

        rounded_star_df['delta_t'] = rounded_star_df.time_int.diff()
        rounded_per_df['delta_t'] = rounded_per_df.time_int.diff()
        rounded_star_df.set_index('time_int', inplace=True, verify_integrity=True)
        rounded_per_df.set_index('time_int', inplace=True, verify_integrity=True)

        avg_len_star.append(rounded_star_df.index.max())
        avg_len_per.append(rounded_per_df.index.max())
        # the length of this array is the number of threads, while the entries correspond to the duration

        # SPEED WITH MOVING AVERAGE
        rounded_star_df['time_int'] = rounded_star_df.index
        rounded_per_df['time_int'] = rounded_per_df.index
        rounded_star_df['delta_n'] = rounded_star_df['number'].diff()
        rounded_per_df['delta_n'] = rounded_per_df['number'].diff()
        rounded_star_df['velocity'] = rounded_star_df['delta_n'] / rounded_star_df['delta_t']
        rounded_per_df['velocity'] = rounded_per_df['delta_n'] / rounded_per_df['delta_t']

        #save csv
        folder = f'../data-tidy/growth_in_time_{var}_{time_interval}' if subreddit == 'aita' else f'../data-tidy/growth_{subreddit}_{var}_{time_interval}'
        # create folder if not existing
        if not os.path.exists(folder):
            os.makedirs(folder)
        #rounded_star_df.to_csv(f'{folder}/star_{str(f)}') # temporarly removed
        #rounded_per_df.to_csv(f'{folder}/per_{str(f)}')


def voters_growth():
    star_df_list = pd.DataFrame()
    per_df_list = pd.DataFrame()
    df = pd.read_csv("../data-tidy/user_edgelist.csv", low_memory=False)
    df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S.%f')
    i = 0
    for subm_id, sub_df in df.groupby('root'):
        i+=1
        sub_df.reset_index(drop=True, inplace=True)
        op=sub_df['to'][0]
        star_df = sub_df[(sub_df['voter']== 1)]
        star_df.sort_values(by='created', inplace=True)
        star_df.reset_index(inplace=True, drop=True)
        per_df = sub_df[(sub_df['voter']== 0)]
        per_df.sort_values(by='created', inplace=True)
        per_df.reset_index(inplace=True, drop=True)
        star_df['timestamp'] = star_df.index
        per_df['timestamp'] = per_df.index
        star_df.sort_values(by='timestamp', inplace=True)
        star_df.reset_index(inplace=True, drop=True)
        per_df.sort_values(by='timestamp', inplace=True)
        per_df.reset_index(inplace=True, drop=True)


        # TIME INTERVALS
        star_df['number'] = star_df.index
        per_df['number'] = per_df.index
        star_df['time_rounded'] = minute_rounder(star_df.created) # rounding to nearest minute
        per_df['time_rounded'] = minute_rounder(per_df.created)
        star_df['time_int'] = star_df['time_rounded'].diff().fillna(timedelta(0)).apply(
            lambda x: x.total_seconds() / 60) # converts the time differences to minutes
        per_df['time_int'] = per_df['time_rounded'].diff().fillna(timedelta(0)).apply(
            lambda x: x.total_seconds() / 60)
        star_df['time_int'] = star_df['time_int'].cumsum() # total time elapsed up to that point
        per_df['time_int'] = per_df['time_int'].cumsum()
        rounded_star_df = star_df.groupby(['time_rounded'])['number', 'time_int'].mean()
        rounded_per_df = per_df.groupby(['time_rounded'])['number', 'time_int'].mean()

        rounded_star_df.set_index('time_int', inplace=True, verify_integrity=True)
        rounded_per_df.set_index('time_int', inplace=True, verify_integrity=True)

        star_df_list = star_df_list.merge(rounded_star_df, how='outer', left_index=True, right_index=True,
                                          suffixes=['', f'{str(i + 1)}'])
        per_df_list = per_df_list.merge(rounded_per_df, how='outer', left_index=True, right_index=True,
                                        suffixes=['', f'{str(i + 1)}'])

    return star_df_list, per_df_list

def avg_subgraphs_growth(cut1, cut2, subreddit, time_interval, edges, errors=True, voters=False):

    if voters:
        star_df_list, per_df_list = voters_growth()
    else:
        star_df_list, per_df_list = subgraphs_growth(subreddit, time_interval, edges=edges)

    star_df_list['time_int'] = star_df_list.index
    per_df_list['time_int'] = per_df_list.index

    star_df_list.reset_index(drop=True, inplace=True)
    per_df_list.reset_index(drop=True, inplace=True)



    star_df_list = star_df_list[(star_df_list.time_int >= cut1) & (star_df_list.time_int < cut2)]
    per_df_list = per_df_list[(per_df_list.time_int >= cut1) & (per_df_list.time_int < cut2)]

    sns.set_theme()
    plt.figure(figsize=(14, 6), dpi=600)
    star_label = 'voters' if voters else 'star'
    per_label = 'non-voters' if voters else 'periphery'
    plt.scatter(star_df_list.time_int, star_df_list['avg'], color='r', label=star_label, s=1)
    plt.scatter(per_df_list.time_int, per_df_list['avg'], color='b', label=per_label, s=1)

    if errors:
        plt.fill_between(star_df_list.time_int, star_df_list.avg.min(), star_df_list.avg.max(), color='r', alpha=0.1)
        plt.fill_between(per_df_list.time_int, per_df_list.avg.min(), per_df_list.avg.max(), color='b', alpha=0.1)

    plt.ylabel('Number of Nodes')
    plt.xlabel('Time (minutes)')
    plt.title(f'subreddit')
    plt.grid(axis='y')
    plt.yticks();
    plt.legend(loc="upper left")

    plt.show()


