import pandas as pd
import numpy as np
from functools import partial



def generate_data(N, P, averages, stds, random_state = None):

    K, H = averages.shape

    base_row, remainder_row = (N // K), (N % K)
    nb_objects =  [base_row + 1] * remainder_row + [base_row] * (K - remainder_row)

    base_col, remainder_col = (P // H), (P % H)
    nb_features =  [base_col + 1] * remainder_col + [base_col] * (K - remainder_col)


    X = pd.DataFrame(np.zeros((N,P)))

    lk = np.cumsum(nb_objects,dtype='int64')
    lk = np.concatenate((np.array([0]),lk))
    labels_objects = np.concatenate([np.repeat(k, nb_objects[k]) for k in range(K)])
    
    lh = np.cumsum(nb_features,dtype='int64')
    lh = np.concatenate((np.array([0]),lh))
    labels_variables = np.concatenate([np.repeat(h, nb_features[h]) for h in range(H)])
   
    np.random.seed(random_state)
    for k in range(K):
        ik = np.arange(lk[k],lk[k+1])
        for h in range(H):
            jh = np.arange(lh[h],lh[h+1])
            X.iloc[ik,jh]= np.random.normal(loc = averages[k,h],scale = stds[k,h], size = ( nb_objects[k], nb_features[h] )) 
    
    return X.to_numpy(),labels_objects.astype('int64'), labels_variables.astype('int64')


averages = np.array([0.0, 1.0, 0.0, 0.0,  2.0, 3.0, 4.0, 5.0,  0.0, 6.0, 0.0, 0.0,0.0, 7.0, 0.0,0.0]).reshape((4,4))
stds = np.array([2.0, 0.5, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 2.0, 0.5, 2.0, 2.0, 2.0, 0.5, 2.0, 2.0]).reshape((4,4))
synthetic_dataset = partial(generate_data, averages = averages, stds = stds)

