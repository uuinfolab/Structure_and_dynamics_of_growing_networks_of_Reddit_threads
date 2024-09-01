from sklearn.calibration import LabelEncoder
from AITA_data import *
from network_analysis import *


''' PROCESS DATA '''

full_path='../data-raw/CSV'
export_path='../data-tidy/processed_CSV'
# process_aita_data(full_path, export_path)
# threads_stats_df(export_path)
# build_edgelists(user=False, comment=True)
# user_nodelist(export_path)
# compute_user_network(draw=False)
### if you donwnload the data from Zenodo, you can skip the previous lines 


''' DATA ANALYSIS '''
# include_struct_prop()
# include_recipr()
### if you donwnload the data from Zenodo, you can skip the previous lines 

static_networks_df=pd.read_csv("./data-tidy/threads_properties.csv", index_col=0)
static_networks_df['periphery_perc'] = 100 - static_networks_df.first_level_users_perc
static_networks_df['avg_text_len'] = static_networks_df[['illegal_NTA_avg_len', 'illegal_YTA_avg_len',
                                           'legal_NTA_avg_len', 'legal_YTA_avg_len', 'unsure_avg_len']].mean(axis=1)
le=LabelEncoder()
static_networks_df['final_judg'] = le.fit_transform(static_networks_df['final_judg'])
print(dict(zip(le.classes_, le.transform(le.classes_))))


vars=['unique_authors_perc', 'num_deleted', 'num_comments',
         'periphery_perc', 'didnt_vote_perc',
         'silent_comments_perc', 'vocal_comments_perc', 'avg_vocality',
         'legal_voters_perc',  # 'illegal_voters_perc',
         'NTA_voters_perc', 'YTA_voters_perc', 'unsure_perc',
         'avg_text_len', 'post_score',  # 'avg_num_comments',
         'comment_frequency', 'thread_duration', 'post_readability', 'personal_pronouns', 'post_sentiment', 'verbs', 'adjectives',
         'post_entropy', 'final_judg']


''' STATISTICAL TESTS'''
# scale_free_test() # generates file "scale_free_test.csv" in "data-tidy" folder
scale_free_analysis()


''' RECIPROCITY AND RESPONSIVENESS '''
# reciprocity_in_time() ### if you donwnload the data from Zenodo, you can skip this line 
responsiveness()


''' GROWING NETWORKS AND SPEED '''

subreddit = 'aita'
path_to_subreddit = None # specify path to data if the subreddit is not 'aita' # please refer to Table 4 in the paper to find the link to download the non 'aita' datasets

# subgraphs_growth_pre(subreddit=subreddit, path_to_subreddit=path_to_subreddit, time_interval='1m', edges=True) # save growing networks in time as csv
# if you donwnload the data from Zenodo, you can skip the previous line

# examples of aita cuts
n1 = 0
n2 = 250

for t in ['1m', '10m', '1h']:
    #star, periphery = subgraphs_growth(subreddit=subreddit, time_interval=t, edges=True, cut1=0, cut2=75)
    avg_subgraphs_growth(cut1=n1, cut2=n2, subreddit='subreddit', time_interval=t, edges=False, errors=False, voters=False) 
    