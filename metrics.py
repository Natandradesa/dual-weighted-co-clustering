import numpy as np
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, silhouette_samples
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import linear_sum_assignment as linear_assignment


def sigma2est(X):
    if type(X) != np.ndarray:
        X = X.to_numpy() 
    X_flat = X.reshape(-1,1) # column vector
    dist = pdist(X_flat, 'sqeuclidean')
    Q90 = np.quantile(dist, [0.9])
    
    if Q90 == 0.0:
        print("Q90 == 0")
        dist = dist[dist > 0.0]

    return (np.quantile(dist, [0.1, 0.9]).sum()/4)

def fuzzy_to_crisp(U):
    if type(U) != np.ndarray:
        U = U.to_numpy() 
    return U.argmax(axis = 1)

def _crisp2fuzzy(P, K):
    N = len(P)
    P_fuzzy = np.zeros((N, K))
    for i in range(N):
        cluster = P[i]
        P_fuzzy[i,cluster] = 1
    return P_fuzzy

def F_measure(y_true,y_pred):
    x = confusion_matrix(y_true,y_pred)
    c,K = x.shape
    n = x.sum()
    n_true = x.sum(axis = 1)
    n_pred = x.sum(axis = 0)
    F_score = np.zeros((c,K))
    for i in range(c):
        for j in range(K):
            F_score[i,j] = (2*x[i,j])/(n_true[i] + n_pred[j])
    return ( (F_score.max(axis = 1) * n_true).sum() )/n

def OERC(y_true, y_pred):
    D = confusion_matrix(y_true, y_pred)
    n = D.sum()
    S = D.max(axis = 0)
    return (1 - (S.sum())/n)

def NMI(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred, average_method ='geometric')

def accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # make a cost matrix
    s = np.max(cm)
    cost = - cm + s
    row_ind, col_ind = linear_assignment(cost)
    total = cm[row_ind,col_ind].sum()
    return (total * 1. / np.sum(cm))

def modified_partition_coefficient(U):
    if type(U) != np.ndarray:
        U = U.to_numpy() 
    n,c = U.shape
    PC = ((U**2).sum())/n
    return 1 - (c/(c-1))*(1-PC)

def partition_entropy_coefficient(U):
    U = U.to_numpy()
    n = U.shape[0]
    M = -(U * np.log(U))
    M = np.where(np.isnan(M),0,M)
    return M.sum()/n

def frigui_index(U,P):

    # U is a fuzzy membership matrix provided by a clustering algorithm
    # P is the real class of the data. A ground truth (a crisp partition).

    N,K = U.shape
    Pf = _crisp2fuzzy(P = P, K = K)

    # coincidence matrices
    cm1 = np.inner(U,U)
    cm2 = np.inner(Pf,Pf)

    # selecting the values of the lower triangle matrix (can be the upper triangle because the matix is symetric)
    idx_lower = np.tril_indices(N, k = -1)
    cm1 = cm1[idx_lower]
    cm2 = cm2[idx_lower]

    NSS = (cm1 * cm2).sum()
    NSD = (cm1 * (1 - cm2) ).sum()
    NDS = ( (1 - cm1) * cm2 ).sum()
    NDD = ( (1 - cm1) * (1 - cm2) ).sum()

    return (NSS + NDD) / (NSS + NSD + NDS + NDD)
    
def hullermeier_index(U,P):
    # U is a fuzzy membership matrix provided by a clustering algorithm
    # P is the real class of the data. A ground truth (a crisp partition).

    N,K = U.shape
    Pf = _crisp2fuzzy(P = P, K = K)

    # matrices of distances
    D1 = cdist(U, U, 'cityblock')/2
    D2 = cdist(Pf, Pf, 'cityblock')/2

    # selecting the values of the lower triangle matrix (can be the upper triangle because the matix is symetric)
    idx_lower = np.tril_indices(N, k = -1)
    D1 = D1[idx_lower]
    D2 = D2[idx_lower]  

    num = np.abs(D1 - D2).sum()
    den = (N/2) * (N-1)
    return 1 - (num/den)
    
def silhouette_indexes(X, U, alpha = 1):
    # This function returns the values of the crisp and fuzzy silhouette average. X
    # X is a dataset and U is a fuzzy partition provided by a clustering algorithm applied to X

    #Getting a crisp partition from the fuzzy partition
    U = U.to_numpy()
    P = U.argmax(axis = 1)

    if len(np.unique(P)) == 1:
        return np.array([np.nan, np.nan])
    
    # getting the crisp silhouette 
    si = silhouette_samples(X, P)
    s_crisp = si.mean()

    # getting the fuzzy silhouette 
    U.sort(axis = 1)
    first_U = U[:,-1] # the first largest membership degree of all objets
    second_U = U[:,-2] # the second largest membership degree of all objets
    diff = (first_U - second_U) ** alpha
    s_fuzzy  = np.average(a = si, weights = diff)
    return np.array([s_crisp, s_fuzzy])
