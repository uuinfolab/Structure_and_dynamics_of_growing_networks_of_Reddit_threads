import os
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
tqdm.pandas()
sns.set_theme()
import warnings
warnings.filterwarnings('ignore')
from utilities import *

''' PROCESSING DATA'''

def process_aita_data(full_path, export_path):
    start = datetime.now()
    for dir_ in os.listdir(full_path):
        if dir_.startswith('Top'):
            for path in os.listdir(f'{full_path}/{dir_}'):
                if path.startswith('1000') and not path.endswith('texts.csv'):

                    top_subm = pd.read_csv(f'{full_path}/{dir_}/{path}', low_memory=False)
                    top_subm['created'] = [datetime.utcfromtimestamp(x) for x in top_subm['created']]
                    top_subm['created'] = pd.to_datetime(top_subm['created'], format='%Y-%m-%d %H:%M:%S.%f')

                    # add missing columns
                    in_18_hours_func(top_subm)
                    top_subm['first_level'] = [1 if x == 0 else 0 for x in top_subm['depth']]
                    top_subm['only_1_vote'] = [1 if x == 1 else 0 for x in top_subm['number_text_flairs']]
                    legal_func(top_subm)
                    vocal_func(top_subm)
                    # dtypes
                    for col in ['in_18_hours', 'first_level', 'legal', 'vocal', 'only_1_vote']:
                        top_subm[col] = top_subm[col].astype('Int64')

                    # adjustments
                    top_subm.loc[top_subm['type'] == 'sub',
                        ['in_18_hours', 'first_level', 'legal', 'vocal', 'only_1_vote']] = np.nan  # not applicable

                    # EXPORT
                    top_subm.to_csv(f'{export_path}/{dir_}_{path}', index=True)
            print('One file processed. Time elapsed: ', datetime.now()-start)
    end = datetime.now()
    print("Time needed to process all data: ", end-start)


def threads_stats_df(export_path):
    final_dict = []

    file_list=[f for f in os.listdir(export_path) if f.endswith('.csv')]
    pbar = tqdm(total=len(file_list))
    for file_name in file_list:
        file_path = os.path.join(export_path, file_name)
        df = pd.read_csv(file_path, low_memory=False)

        # adjust types
        df.text_flair_list = df.text_flair_list.astype(str)
        df.text_flair.fillna('', inplace=True)
        df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S.%f')

        # disentangle multiple posts in the same csv file
        for subm_id, sub_df in df.groupby('submission_id'):
            if sub_df.type.nunique() != 1:
                print('ERROR: in submissions considered', file_name, subm_id)
                raise ValueError

            # dtypes
            for col in ['in_18_hours', 'first_level', 'legal', 'vocal', 'only_1_vote']:
                sub_df[col] = sub_df[col].astype('Int64')

            # dictionaries definition to avoid keyerrors
            vote_len_dict = {('NTA', 0): 0, ('NTA', 1): 0, ('YTA', 0): 0, ('YTA', 1): 0, ('', 0): 0}
            vote_dict = {('NTA', 0): 0, ('NTA', 1): 0, ('YTA', 0): 0, ('YTA', 1): 0, ('', 0): 0}
            vote_voc_dict = {('NTA', 0): 0, ('NTA', 1): 0, ('YTA', 0): 0, ('YTA', 1): 0, ('', 0): 0, ('', 1): 0}
            first_lev_users_dict = {key: 0 for key in [0, 1]}
            first_lev_rule_dict = {key: 0 for key in [(0, 0), (1, 0), (1, 1)]}
            time_rule_dict = {key: 0 for key in [(0, 0), (1, 0), (1, 1)]}
            only_one_rule_dict = {key: 0 for key in [(0, 0), (1, 0), (1, 1)]}
            vocal_pos_dict = {key: 0 for key in [(0, 0), (0, 1), (1, 0), (1, 1)]}
            score_voc_dict = {key: 0 for key in [0, 1]}
            score_leg_dict = {key: 0 for key in [0, 1]}
            score_pos_dict = {key: 0 for key in [0, 1]}

            ### compute entropy of votes
            post_entropy = compute_post_entropy(sub_df)

            ### count voters and legals
            legal_voters = len(sub_df[sub_df['legal'] == 1])
            non_legal_voters = len(sub_df[sub_df['legal'] == 0])
            total_voters = len(sub_df[sub_df['voter'] == 1])
            didnt_vote = len(sub_df[sub_df['legal'].isnull()])
            # check correctedness
            if total_voters != legal_voters + non_legal_voters or didnt_vote != len(sub_df[sub_df['voter'] == 0]):
                print('ERROR in computation: ', file_name, subm_id)
                raise ValueError
            NTA_voters = len(sub_df[sub_df.text_flair=='NTA'])
            YTA_voters = len(sub_df[sub_df.text_flair=='YTA'])
            unsure_voters = len(sub_df[(sub_df.number_text_flairs>1) & (sub_df.text_flair.isnull())])

            ### count legality and vocality

            silent = len(sub_df[sub_df.vocal == 0])
            vocal = len(sub_df[sub_df.vocal == 1])
            vote_dict.update(sub_df.groupby(['text_flair', 'legal']).legal.count().to_dict())
            vote_len_dict.update(sub_df.groupby(['text_flair', 'legal']).token_count.mean().round(2).to_dict())
            vote_voc_dict.update(sub_df.groupby(['text_flair', 'vocal']).vocal.count().to_dict())
            vocal_pos_dict.update(sub_df.groupby(['vocal', 'first_level']).vocal.count().to_dict())
            score_voc_dict.update(sub_df.groupby('vocal').score.mean().round(2).to_dict())
            score_leg_dict.update(sub_df.groupby('legal').score.mean().round(2).to_dict())
            score_pos_dict.update(sub_df.groupby('first_level').score.mean().round(2).to_dict())

            # which of the three rules is not respected (i.e. why illegal)
            #subsubdf = sub_df[sub_df.voter==1] # not needed
            first_lev_users_dict.update(sub_df.groupby('first_level').first_level.count().to_dict())
            first_lev_rule_dict.update(sub_df.groupby(['first_level', 'legal']).legal.count().to_dict())
            time_rule_dict.update(sub_df.groupby(['in_18_hours', 'legal']).legal.count().to_dict())
            only_one_rule_dict.update(sub_df.groupby(['only_1_vote', 'legal']).legal.count().to_dict())

            # todo: we miss author flair in karolina data
            subsubdf= sub_df[sub_df.type=='com']
            try:
                sub_sd_dict = {'filename': file_name, 'subm_id': subm_id, 'final_judg': df[df.id==subm_id].link_flair_text.values[0],
                                'number_of_users': len(sub_df.author.unique())-1,
                                'unique_authors_perc': perc(len(sub_df.author.unique())-1, len(sub_df[~sub_df.author.isna()].author)), # excluding deleted
                                'num_deleted': perc(len(sub_df[sub_df.author.isna()]), len(sub_df.author)), #len(sub_df[sub_df.deleted==True])
                                'num_comments': df[df.id==subm_id].comment_count.values[0], 'total_voters_perc': perc(total_voters, len(sub_df)), 'didnt_vote_perc': perc(didnt_vote, len(sub_df)),

                                ### LEGALITY ###
                                'legal_voters_perc': perc(legal_voters, total_voters), 'non_legal_voters_perc': perc(non_legal_voters, total_voters),
                                'broke_1st_lev_rule_perc': perc(first_lev_rule_dict.get((0, 0), 0), non_legal_voters),
                                'broke_18h_rule_perc': perc(time_rule_dict.get((0, 0), 0), non_legal_voters),
                                'broke_only1vote_rule_perc': perc(only_one_rule_dict.get((0, 0), 0), non_legal_voters),

                                ### VOTE ###
                                'NTA_voters_perc': perc(NTA_voters, total_voters), 'YTA_voters_perc': perc(YTA_voters, total_voters), 'unsure_voters_perc': perc(unsure_voters, total_voters),
                                'unsure_perc': perc(vote_dict[('', 0)], total_voters),
                                ### ENTROPY AND VOCALITY ###
                                'post_entropy': post_entropy, 'silent_comments_perc':perc(silent, total_voters), 'vocal_comments_perc':perc(vocal, total_voters), 'avg_vocality':sub_df['vocal'].mean(),

                                ### POSITION AND SCORE ###
                                'first_level_users_perc':perc(first_lev_users_dict[(1)], len(sub_df)), # star dimension
                                'avg_text_len':sub_df['token_count'].mean(), 'avg_comm_score':sub_df['score'].mean(),
                                'post_score':df[df.id==subm_id].score.values[0],
                                # 'score_vocal':score_voc_dict[(1)], 'score_legal':score_leg_dict[(1)], 'score_star':score_pos_dict[(1)],
                                # 'score_silent': score_voc_dict[(0)], 'score_illegal': score_leg_dict[(0)], 'score_per': score_pos_dict[(0)],

                                ### TIME ###
                                'comment_frequency': df.groupby(df.created.dt.minute).count().mean()[0], # todo: modificato 11 gennaio, da runnare e verificare
                                'thread_duration': (sub_df.created.max() - sub_df.created.min()) / np.timedelta64(1, 'h'), #  measured in hours

                                ### LANGUAGE FEATURES ###
                                'post_readability': df[df.id==subm_id].readability.values[0], 'personal_pronouns':df[df.id==subm_id].personal_pronouns.values[0],
                                'post_sentiment': df[df.id==subm_id].sentiment_compound_VADER.values[0],
                                'verbs': df[df.id==subm_id].verbs.values[0], 'adjectives': df[df.id==subm_id].adjectives.values[0],

                                ### MIXED ###
                                'vocal_star':perc(vocal_pos_dict.get((1,1), 0), total_voters), 'vocal_per':perc(vocal_pos_dict.get((1,0), 0), total_voters),
                                'illegal_NTA_voters_perc':perc(vote_dict[('NTA', 0)]+vote_dict.get(('NAH', 0), 0), total_voters),
                                'illegal_YTA_voters_perc':perc(vote_dict[('YTA', 0)]+vote_dict.get(('ESH', 0), 0), total_voters),
                                'legal_NTA_voters_perc':perc(vote_dict[('NTA', 1)]+vote_dict.get(('NAH', 1), 0), total_voters),
                                'legal_YTA_voters_perc': perc(vote_dict[('YTA', 1)]+vote_dict.get(('ESH', 1), 0), total_voters),
                                'illegal_NTA_avg_len': vote_len_dict[('NTA', 0)], 'illegal_YTA_avg_len': vote_len_dict[('YTA', 0)],
                                'legal_NTA_avg_len': vote_len_dict[('NTA', 1)], 'legal_YTA_avg_len': vote_len_dict[('YTA', 1)],
                                'unsure_avg_len':vote_len_dict[('', 0)],
                                'NTA_silent_perc':perc(vote_voc_dict[('NTA', 0)], total_voters), 'YTA_silent_perc':perc(vote_voc_dict[('YTA', 0)], total_voters),
                                'NTA_vocal_perc':perc(vote_voc_dict[('NTA', 1)], total_voters), 'YTA_vocal_perc':perc(vote_voc_dict[('YTA', 1)], total_voters),
                                'unsure_silent_perc':perc(vote_voc_dict[('', 0)], total_voters), 'unsure_vocal_perc':perc(vote_voc_dict[('', 1)], total_voters)}
            except KeyError as e:
                print('Error in file', file_path, e)
            s = pd.Series(sub_sd_dict)
            final_dict.append(s)

        pbar.update(n=1)

    static_networks_df = pd.DataFrame(final_dict)
    static_networks_df.to_csv("../data-tidy/threads_stats.csv", index=True)


''' BUILD COMMENT TREE EDGE TABLE '''

def build_edgelists(export_path, user=True, comment=True):
    print('Executing function build_edgelists')
    comm_tree_edgelist_merged = pd.DataFrame()
    user_edgelist_merged = pd.DataFrame()

    file_list = [f for f in os.listdir(export_path) if f.endswith('.csv')]
    for file_name in file_list:
        file_path = os.path.join(export_path, file_name)
        df = pd.read_csv(file_path, low_memory=False)
        df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S.%f')

        comm_tree_edgelist = df[['submission_id', 'id', 'parent_id', 'created', 'voter', 'text_flair', 'depth']]
        comm_tree_edgelist.rename(columns={'submission_id': 'root', 'id': 'from', 'parent_id':'to'}, inplace=True)
        comm_tree_edgelist['to']=comm_tree_edgelist['to'].str.split('_').str[1]
        comm_tree_edgelist_merged = pd.concat([comm_tree_edgelist_merged, comm_tree_edgelist])
        if user:
            author_comment = df[['id', 'author']]
            author_comment.set_index('id', inplace=True)
            author_comment = author_comment.author.to_dict()
            user_edgelist = comm_tree_edgelist.copy()
            user_edgelist['from'] = user_edgelist['from'].map(author_comment)
            user_edgelist['to'] = user_edgelist['to'].map(author_comment)
            user_edgelist_merged = pd.concat([user_edgelist_merged, user_edgelist])
    if comment:
        comm_tree_edgelist_merged.to_csv("../data-tidy/comm_tree_edgelist_withvote.csv", index=False)
    if user:
        user_edgelist_merged.to_csv("../data-tidy/user_edgelists.csv", index=False)
        print(user_edgelist_merged.info())

def user_nodelist(export_path):
    entropy_df_merged=pd.DataFrame()

    file_list = [f for f in os.listdir(export_path) if f.endswith('.csv')]
    pbar = tqdm(total=6366)
    for file_name in file_list:
        file_path = os.path.join(export_path, file_name)
        df = pd.read_csv(file_path, low_memory=False)

        # adjust types
        df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S.%f')

        for subm_id, sub_df in df.groupby('submission_id'): # groupby excludes NaNs so OP are not counted in the authors list
            if sub_df.type.nunique() != 1:
                print('ERROR: in submissions considered', file_name, subm_id)
                raise ValueError

            final_judg=df[df.id==subm_id].link_flair_text.values[0]
            entropy_df = compute_entropy_in_time(sub_df, final_judg)
            '''
            node_df = pd.DataFrame()
            node_df[['author', 'entering_time']] = sub_df[['created', 'author']].groupby('author').min().reset_index()
            node_df['post'] = subm_id
            node_df['voter_on_avg'] = sub_df[['voter', 'author']].groupby('author').mean().reset_index()['voter']
            node_df['legal_on_avg'] = sub_df[['legal', 'author']].groupby('author').mean().reset_index()['legal']
            node_df['vocal_vot_on_avg'] = sub_df[['vocal', 'author']].groupby('author').mean().reset_index()['vocal']
            node_df['vocal_in_general'] = sub_df[['readability_FK_GL', 'author']].groupby('author').mean().reset_index()['readability_FK_GL'] # average readability of their comments
            node_df['num_comments'] = sub_df[['id', 'author']].groupby('author').count().reset_index()['id']
            node_df['avg_text_len'] = sub_df[['token_count', 'author']].groupby('author').mean().reset_index()['token_count']
            node_df['most_freq_vote'] = sub_df[['text_flair', 'author']]['text_flair'].agg(pd.Series.mode)
            node_df['avg_sentiment'] = sub_df[['sentiment_compound_VADER', 'author']].groupby('author').mean().reset_index()['sentiment_compound_VADER']
            #node_df['OP'] = False
            #node_df.loc[node_df['post'].isna(), 'OP'] = True

            node_df = node_df[~node_df['author'].isna()] # remove deleted users
            '''
            #user_nodelist_merged = pd.concat([user_nodelist_merged, node_df])
            entropy_df_merged = pd.concat([entropy_df_merged, entropy_df])
            pbar.update(n=1)
            #for subm_id, sub_df in user_nodelist_merged.groupby('post'):
                #sub_df.to_csv(f"../data-tidy/user_networks_nodelist/user_nodelist_post_{str(subm_id)}.csv", index=False)

            #print(user_nodelist_merged.info())
    #user_nodelist_merged.to_csv(f"../data-tidy/user_networks_nodelist/user_nodelist.csv", index=False)
    entropy_df_merged.to_csv(f"../data-tidy/entropy_in_time.csv", index=False)


''' COMPUTE RESPONSIVENESS'''

def responsiveness():
    # check if file not already exists
    if not os.path.exists('./data-tidy/comm_tree_edgelist_responsiveness.csv'):
        print('Computing responsiveness...')
        df = pd.read_csv('./data-tidy/comm_tree_edgelist_withvote.csv')
        df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S.%f')
        parent_timestamps = df.set_index('from')['created'].to_dict()

        def calculate_time_difference(row):
            parent_id = row['to']
            if parent_id is not None and parent_id in parent_timestamps:
                return row['created'] - parent_timestamps[parent_id]
            else:
                return pd.NaT

        df['time_diff'] = df.apply(calculate_time_difference, axis=1)
        df.to_csv('./data-tidy/comm_tree_edgelist_responsiveness.csv', index=True)
    else:
        print('Responsiveness already computed. Uploading corresponding file...')
        df = pd.read_csv('./data-tidy/comm_tree_edgelist_responsiveness.csv')
        df['time_diff'] = pd.to_timedelta(df['time_diff'], errors='coerce')
        df['time_diff'] = df['time_diff'].dt.total_seconds()

        # Filter time differences within [μ - nσ, μ + nσ]
        n = 1
        lower_bound = df['time_diff'].mean() - n * df['time_diff'].std()
        upper_bound = df['time_diff'].mean() + n * df['time_diff'].std()
        filtered_df = df[(df['time_diff'] >= lower_bound) & (df['time_diff'] <= upper_bound)]
        print(len(filtered_df))
        filtered_df = filtered_df[filtered_df['time_diff'].notnull() & filtered_df['time_diff'] > 0]
        print(len(filtered_df))
        print("Mean:", df['time_diff'].mean() )
        print("Standard Deviation:", df['time_diff'].std())

        filtered_df['inverse_time_difference'] = 1 / filtered_df['time_diff']

        filtered_df['star'] = np.nan
        roots = filtered_df['root'].unique().tolist()
        filtered_df.loc[filtered_df['to'].isin(roots), 'star']= 'Star' # ops and users answering to them
        filtered_df.loc[~filtered_df['to'].isin(roots) & ~filtered_df['to'].isna(), 'star'] = 'Periphery' # all the others

        
        ### BOXPLOT

        plt.figure(figsize=(10, 6))
        #sns.pairplot(filtered_df, diag_kind='kde')
        sns.boxplot(filtered_df, x='star', y='time_diff', hue='voter')
        plt.yscale('log')
        plt.legend(title='Voter', loc='upper right')
        plt.ylabel('Responsiveness (seconds)')
        plt.xlabel('')
        plt.show()
        means = filtered_df.groupby(['star', 'voter'])['time_diff'].median().reset_index()
        # print(means)

