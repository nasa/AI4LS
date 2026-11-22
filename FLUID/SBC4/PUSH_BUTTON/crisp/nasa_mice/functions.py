import pandas as pd
import numpy as np
# try:
#     import rbo
# except:
#     ! pip install rbo
import rbo

def generate_mouse_human_experiment(filepath='data/combined_mouse_human_data.pkl',
                                    human_count = 125,
                                    mice_count = 0,
                                    seed = None,
                                    pickle_directory = None,
                                    organism_column = 'organism'):
    file = pd.read_pickle(filepath)
    mouse_sample = file[file[organism_column] == 'Mus musculus'].sample(n=mice_count, random_state = seed)
    human_sample = file[file[organism_column] == 'Homo sapiens'].sample(n=human_count, random_state = seed)
    human_sample = human_sample.append(mouse_sample).sample(frac=1, random_state = seed)
    if pickle_directory == None:
        return human_sample
    else:
        human_sample.to_pickle(pickle_directory+'/humans_'+str(human_count)+'_mice_'+str(mice_count)+'_seed_'+str(seed)+'.pkl')

        
class data_comparisons:
    def __init__(self, base, comparison, model_idx):
        self.base_frame_untrimmed = get_rank_frame(base, model_idx)
        self.comparison_frame_untrimmed  = get_rank_frame(comparison, model_idx)
        self.base_frame, self.comparison_frame = trim_frames(self.base_frame_untrimmed,self.comparison_frame_untrimmed)
        self.base_rank = rank(self.base_frame)
        self.comparison_rank = rank(self.comparison_frame)
        self.metrics = {'top_10_overlap_percent': top_n_overlap_percent(self.base_rank,self.comparison_rank,n=10),
                        'top_20_overlap_percent': top_n_overlap_percent(self.base_rank,self.comparison_rank,n=20),
                        'top_50_overlap_percent': top_n_overlap_percent(self.base_rank,self.comparison_rank,n=50),
                        'ranked_bias_overlap': ranked_bias_overlap(self.base_rank,self.comparison_rank),
                        'kendall_tau': kendall_tau(self.base_rank,self.comparison_rank),
                        'cosine_similarity': cosine_similarity(self.base_frame, self.comparison_frame)
                       }

        
def get_rank_frame(json, model_idx):
    features = json['results'][model_idx]['features']
    coeff = json['results'][model_idx]['coefficients']
    df = pd.DataFrame([features,coeff]).T
    df.columns = ['features','coefficients']
    df['rank'] = abs(df['coefficients'])
    df = df.sort_values(['rank'],ascending = False)
    return df

def trim_frames(df1,df2):
    size_cap = min(len(df1),len(df2))
    df1 = df1.iloc[:size_cap]
    df2 = df2.iloc[:size_cap]
    return df1, df2

def rank(df):
    return list(df['features'])

def ranked_bias_overlap(base_rank, comparator_rank):
    return rbo.RankingSimilarity(base_rank, comparator_rank).rbo()

def top_n_overlap_percent(base_rank, comparator_rank, n=10):
    return len(set(base_rank[0:50]).intersection(set(comparator_rank[0:n]))) / n

from scipy.stats import kendalltau
def kendall_tau(base_rank,comparator_rank):
    return kendalltau(base_rank, comparator_rank)

from scipy.spatial.distance import cosine
def cosine_similarity(base_frame, comparator_frame):
    return 1 - cosine(base_frame['coefficients'].astype(float).to_numpy(),
                  comparator_frame['coefficients'].astype(float).to_numpy())

