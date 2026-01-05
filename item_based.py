import pandas as pd
import numpy as np
import pickle


# load data
ratings = pd.read_csv(r'data/ratings_small.csv')

result = []
seeds = [15, 137, 233, 589, 768]
for seed in seeds:
    print(seed)
    # random seed
    np.random.seed(seed)

    # split data
    split = np.random.rand(len(ratings)) < 0.8
    train = ratings[split]
    test = ratings[~split]
    
    # rating matrix
    ratingsmat = train.pivot(index='userId', columns='movieId', values='rating')
    userid = ratingsmat.index
    movieid = ratingsmat.columns
    
    # normalized
    ratingsnorm = ratingsmat-ratingsmat.mean()
    
    # similarity
    def cosinesimilarity(x,y):
        # cosine similarity
        inner = (x*y).dropna()
        if inner.shape[0]==0:
            return np.nan
        normx = np.linalg.norm(x.dropna())
        if normx==0:
            return np.nan
        normy = np.linalg.norm(y.dropna())
        if normy==0:
            return np.nan
        
        cos = inner.sum()/normx/normy
        return cos
    
    # test movies
    testmovies = list(set(movieid).intersection(set(test.movieId)))
    nmovie = len(testmovies)
    """
    simmat = np.empty((nmovie,nmovie))
    simmat.fill(np.nan)
    for i in range(nmovie-1):
        for j in range(i+1,nmovie):
            idi,idj = testmovies[i],testmovies[j]
            sim = cosinesimilarity(ratingsnorm[idi], ratingsnorm[idj])
            simmat[i,j] = sim
            simmat[j,i] = sim
    pickle.dump(simmat,open(r'data/simmat-%s.pkl' %seed,'wb'))
    """
    simmat = pickle.load(open(r'data/simmat-%s.pkl' %seed,'rb'))
    
    # similarity dataframe
    simdf = pd.DataFrame(index=testmovies,columns=testmovies,data=simmat)
    
    # recommendation
    forecast = test.copy()
    forecast['frating'] = np.nan
    for i in test.index:
        u,m = test.loc[i]['userId'],test.loc[i]['movieId']
        ru = ratingsmat.loc[u,:]
        if m in simdf.columns:
            sim = simdf[m]
            rate = pd.concat([ru.rename('r'),sim.rename('s')],axis=1).dropna()
            rate = rate[rate['s']>0]
            rate['as'] = rate['s']
            """
            rate['as'] = np.exp(rate['s'])
            rate['as'] = rate['as']/rate['as'].sum()
            """
            if rate.shape[0] == 0:
                rate = np.nan
            else:
                rate = (rate['r']*rate['as']).sum()/rate['as'].sum()
        else:
            rate = np.nan
        forecast.loc[i,'frating'] = rate
        
    # RMSE
    #rmse = np.sqrt(sum((forecast['rating'] - forecast['frating']).dropna()**2)/(forecast.shape[0]-forecast['frating'].isna().sum()))
    diff = (forecast['rating'] - forecast['frating']).dropna()
    mae = np.abs(diff).mean()
    with open('result %s.txt' %seed,'w') as f:
        f.write(str(mae))
    
    result.append(mae)
result = pd.DataFrame({'seed':seeds,'MAE':result})
result.to_excel('result.xlsx')
