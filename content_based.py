import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# seed list
seeds = [15, 137, 233, 589, 768]

for seed in seeds:
    # random seeds
    np.random.seed(seed)
        
    # loading the dataset
    metadata = pd.read_csv("movies_metadata.csv")
    credit = pd.read_csv("credits.csv")
    keywords = pd.read_csv("keywords.csv")
    links = pd.read_csv("links_small.csv")
    ratings = pd.read_csv("ratings_small.csv")
        
    # extract the updated metadata based on links_small csv file
    links = links[links['tmdbId'].notna()]
    links['tmdbId'] = links['tmdbId'].astype('int')
    slinks = links['tmdbId']   
    metadata = metadata.drop([19730, 29503,35587])
    metadata['id'] = metadata['id'].astype('int')
        
    # create a dictionary about the relationship between ids and tmdbId
    # add tmdbId into the rating file
    linksdict = dict()
    for index, row in links.iterrows():
        linksdict[row['movieId']] = row['tmdbId']    
    new_value = list()
    for index, row in ratings.iterrows():   
        if row['movieId'] not in linksdict.keys():
            new_value.append(-1)
        else:
            new_value.append(linksdict[row['movieId']])           
    value_ser = pd.Series(data = new_value)
    ratings.insert(2, 'tmdbId', value_ser)
       
    # split rating data
    split = np.random.rand(len(ratings)) < 0.8
    train = ratings[split]
    test = ratings[~split]
        
    # merge credit and keywords into metadata
    metadata = metadata.merge(credit, on='id')
    metadata = metadata.merge(keywords, on='id')
    umetadata = metadata[metadata['id'].isin(slinks)]
       
    # delete the lines from train and test dataset w/o tmdbId and metadata
    train = train.drop(train[train['tmdbId'] == -1].index)
    test = test.drop(test[test['tmdbId'] == -1].index)   
    sslinks = metadata['id']
    train = train[train['tmdbId'].isin(sslinks)]
    test = test[test['tmdbId'].isin(sslinks)]
    
    
    # the follow code block refers to the following sample code
    # https://www.kaggle.com/code/rounakbanik/movie-recommender-systems
    
    # ------------------------------------------------------------------------------------------------------------------ 
    # extract director from crew and pre-process the data
    umetadata['cast'] = umetadata['cast'].apply(literal_eval)
    umetadata['crew'] = umetadata['crew'].apply(literal_eval)
    umetadata['keywords'] = umetadata['keywords'].apply(literal_eval)
    
    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan
    umetadata['director'] = umetadata['crew'].apply(get_director)
    umetadata['director'].fillna('', inplace=True)
    
    umetadata['cast'] = umetadata['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    umetadata['cast'] = umetadata['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
    umetadata['keywords'] = umetadata['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    
    umetadata['cast'] = umetadata['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    umetadata['director'] = umetadata['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    umetadata['cast'] = umetadata['cast'].apply(lambda x: ' '.join(x))
    umetadata['keywords'] = umetadata['keywords'].apply(lambda x: ' '.join(x))
        
    # concernate these four into the full features and compute the similarity
    umetadata['con'] = umetadata['keywords'] + umetadata['cast'] + umetadata['genres'] + umetadata['director']
    count = CountVectorizer(analyzer = 'word',ngram_range = (1, 2),min_df = 0, stop_words = 'english')
    count_matrix = count.fit_transform(umetadata['con'])
    
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    umetadata = umetadata.reset_index()
    titles = umetadata['title']
    indices = pd.Series(umetadata.index, index = umetadata['id'])
    indices = indices[~indices.index.duplicated(keep='first')]
    # ------------------------------------------------------------------------------------------------------------------
    
    # build the dict to represent the ratings from each user
    R_ui = dict()
    for index, row in train.iterrows():
        if row['userId'] not in R_ui:
            temp = {}
            temp[row['tmdbId']] = row['rating']
            R_ui[row['userId']] = temp
        else:
            R_ui[row['userId']][row['tmdbId']] = row['rating']
    
    # predict the ratings
    def content_based_re(user_u, item_i):
        numerator, denominator = 0, 0
        I = R_ui[user_u].keys()
    
        for i in I:
            if i != item_i:
                idx1, idx2 = indices[i], indices[item_i]
                sub_cos = cosine_sim[idx1, idx2]
                numerator += sub_cos * R_ui[user_u][i]
                denominator += abs(sub_cos)
        return numerator/denominator
    
    # calculate the MAE value
    P_ui_list = list()
    for index, row in test.iterrows():
        P_ui = content_based_re(row['userId'], row['tmdbId'])
        P_ui_list.append(P_ui)
    
    R_ui_list = test['rating'].tolist()  
    P_ui_list = np.array(P_ui_list) 
    R_ui_list = np.array(R_ui_list) 
    err_list = np.abs(P_ui_list - R_ui_list)
    err_list = err_list[~np.isnan(err_list)]
    mae = np.mean(err_list)
    print("The MAE of seed {} is {}.".format(seed, mae))