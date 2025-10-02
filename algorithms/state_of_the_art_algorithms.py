import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.utils import check_array, check_random_state
from warnings import warn


class BaseFuzzy():

    def random_U(self, shape, random_state = None):
    
        if type(shape) == int:
            np.random.seed(random_state)
            h = np.random.random(shape)
            return h/h.sum()
        else:
            np.random.seed(random_state)
            h = np.random.random(shape)
            return h/h.sum(axis= 1).reshape(-1,1) 
    
    def initial_cocluster_prototypes(self, X, K, H):
        N,P = X.shape
        Xu = np.unique(X)
        lu = len(Xu)
        tp = K*H
        if lu > tp:
            return np.random.choice(Xu, tp, replace = False).reshape((K,H))
        else:
            Z = np.random.choice(K, N, replace = True)
            W = np.random.choice(H, P, replace = True)
            G = np.zeros((K,H))
            for k in range(K):
                for h in range(H):
                    G[k,h] = np.mean(X[Z == k][:,W == h])
            return G



    def dist_array(self,X, G):
        '''
        X is the dataset
        G is the prototypes of the co-clusters
        '''
        N, P = X.shape
        K, H = G.shape
        D = np.zeros((K, H, N, P))
        for k in range(K):
            for h in range(H):
                D[k,h] = (X - G[k,h]) ** 2    
        return D


    def getU_co_clustering(self, D, U, V, m, n):
        '''
        D is the array of the distances between the dataset and the prototype matrix
        U is the fuzzy partiton of objects into K cluster that will be updated
        V is the fuzzy partiton of variables into H cluster
        m and n are the fuzzyness parameters
        '''

        Vn = V**n
        Vnt = np.transpose(Vn)
        exponent = (1/(m-1))
        K,N = D.shape[0], D.shape[2]
        for i in range(N):
            Di = D[:,:,i,:] 
            Dik = ((Vnt*Di).sum(axis = (1,2))) 
            inv_dk = (1/Dik) ** exponent
            idx_inf =  np.where(np.isinf(inv_dk))[0]
            n_inf = len(idx_inf)
            if n_inf < K:
                idx = np.where(~np.isinf(inv_dk))[0] # values different of inf
                den = inv_dk[idx].sum()
                if den > 0.0 and den != np.inf:
                    if n_inf == 0:
                        U[i] = inv_dk/inv_dk.sum()
                    else:
                        inv_dk_new = inv_dk[idx]
                        sum_previous = U[i,idx_inf].sum()
                        const = (1 - sum_previous)
                        if const >= 0.0:
                            U[i,idx] = const * (inv_dk_new/inv_dk_new.sum())
                        else:
                            U[i,idx] = 0.0
        return U 

    def getV_co_clustering(self, D, U, V, m, n):
        '''
        D is the array of the distances between the dataset and the prototype matrix
        U is the fuzzy partiton of objects into K cluster
        V is the fuzzy partiton of variables into H cluster that will be updated
        m and n are the fuzzyness parameters
        '''
        Um = U**m
        Umt = np.transpose(Um)
        exponent = (1/(n-1))
        H,P = D.shape[1], D.shape[3]
        Dm = np.moveaxis(D,0,1)
        for j in range(P):
            Dj = Dm[:,:,:,j] 
            Djh = ((Umt*Dj).sum(axis = (1,2))) 
            inv_dh = (1/Djh) ** exponent
            idx_inf =  np.where(np.isinf(inv_dh))[0]
            n_inf = len(idx_inf)
            if n_inf < H:
                idx = np.where(~np.isinf(inv_dh))[0]
                den = inv_dh[idx].sum()
                if den > 0.0 and den != np.inf:
                    if n_inf == 0:
                        V[j] = inv_dh/inv_dh.sum()
                    else:
                        inv_dk_new = inv_dh[idx]
                        sum_previous = V[j,idx_inf].sum()
                        const = (1 - sum_previous)
                        if const >= 0.0:
                            V[j,idx] = const * (inv_dk_new/inv_dk_new.sum())
                        else:
                            V[j,idx] = 0.0
        return V

    def computeJ_co_clustering(self, D, Um, Vn):
        '''
        D is the array of the distance between the dataset and the prototype matrix
        Um is the fuzzy matrix of the objects into K cluster raised to the m
        Vn is the fuzzy matrix of the variables into H cluster raised to the n
        '''
        K,H = D.shape[:2]
        Jc = 0
        for k in range(K):
            uk = (Um[:,k]).reshape((-1,1))
            for h in range(H):
                vh = (Vn[:,h]).reshape((1,-1))
                Dkh = uk * D[k,h] * vh
                Jc += Dkh.sum()
        return Jc

    def isdecreasing(self, J):
        J_sorted = J.copy()
        J_sorted.sort(reverse = True)
        if J_sorted == J:
            return True
        else:
            return False


class FuzzyDoubleKmeans(BaseFuzzy):

    def __init__(self, K, H, m, n, epsilon = 10e-6, max_iter = 100, n_init = 1 ):
        self.K = K
        self.H = H
        self.m = m
        self.n = n
        self.epsilon = epsilon
        self.T = max_iter
        self.n_init = n_init

        self.U = None
        self.V = None
        self.G = None
        self.J = np.inf
        self.Jlist = None
        self.iterations = None

    def fit(self, X, random_state = 0):

        if type(X) != np.ndarray:
            X = X.to_numpy()

        random_state = check_random_state(random_state)

        check_array(X, accept_sparse = False, dtype="numeric", order=None,
                    copy=False, force_all_finite=True, ensure_2d=True,
                    allow_nd=False, ensure_min_samples = self.K,
                    ensure_min_features = self.H, estimator=None)

        X = X.astype(float)

        J = self.J
        Jlist = self.Jlist
        U = self.U
        V = self.V
        G = self.G
        iterations = self.iterations

        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for seed in seeds:
            self.__fit_single(X, seed)
            if np.isnan(self.J):
                raise ValueError("matrix may contain negative or unexpected NaN values")
            
            if (self.J < J):
                J = self.J
                Jlist = self.Jlist
                U = self.U
                V = self.V
                G = self.G
                iterations = self.iterations
                
        self.J = J
        self.Jlist = Jlist
        self.U = U
        self.V = V
        self.G = G
        self.iterations = iterations

        if super().isdecreasing(self.Jlist) == False:
            warn('Objective function did not converge')

        return self

    def __prototypes_FDK(self, X, Um, Vn, G):
        '''
        X is the dataset 
        Um is the fuzzy matrix of the objects into K cluster raised to the m
        Vn is the fuzzy matrix of the variables into H cluster raised to the n
        G is the current prototype matrix that will be updated
        ld is the smallest possible value for a denominator. Used to avoid indeterminacy
        '''
        lowest_denominator = 10**(-10)
        K,H = G.shape
        G_new = np.zeros((K,H))
        for k in range(K):
            uk = (Um[:,k]).reshape((-1,1))
            for h in range(H):
                vh = (Vn[:,h]).reshape((1,-1))
                wh = (uk * np.full_like(X,1) * vh)
                if wh.sum() > lowest_denominator:
                    G_new[k,h] = np.average(a = X, weights = wh )
                else:
                    G_new[k,h] = G[k,h]            
        return G_new

    def __fit_single(self, X, random_state = 0):  
        
        if type(X) != np.ndarray:
            X = X.to_numpy()
        X = X.astype('float64')
        N,P = X.shape



        # Inicialization step
        np.random.seed(random_state)
        G = super().initial_cocluster_prototypes(X = X, K = self.K, H = self.H)
        #G = np.random.choice(np.unique(X), self.K*self.H, replace= False).reshape((self.K,self.H))
        U = super().random_U((N,self.K),random_state)
        V = super().random_U((P,  self.H),random_state)
        Um = U**self.m
        Vn = V**self.n
        J = 0
        Jlist = []

        # iterative step
        t = 1
        while True:
            G = self.__prototypes_FDK(X, Um, Vn, G)
            D = super().dist_array(X, G)
            U = super().getU_co_clustering(D, U, V, self.m, self.n)
            V = super().getV_co_clustering(D, U, V, self.m, self.n)
    
            Um = U**self.m
            Vn = V**self.n
            Jcurr = J
            J = super().computeJ_co_clustering(D, Um, Vn)
            Jlist.append(J)

            if np.abs(Jcurr - J) < self.epsilon or t > self.T:
                break
            else:
                t = t + 1

        self.U = U
        self.V = V
        self.G = G
        self.J = J
        self.Jlist = Jlist
        self.iterations = t

        if super().isdecreasing(self.Jlist) == False:
            warn('Objective function did not converge')

        return self


class WFDK(BaseFuzzy):

    def __init__(self, K, H, m, n, gamma, epsilon = 10e-6, max_iter = 100, n_init = 1 ):
        self.K = K
        self.H = H
        self.m = m
        self.n = n
        self.gamma = gamma
        self.epsilon = epsilon
        self.T = max_iter
        self.n_init = n_init

        self.U = None
        self.V = None
        self.G = None
        self.weights = None
        self.J = np.inf
        self.Jlist = None
        self.iterations = None

    def fit(self, X, random_state = 0):

        if type(X) != np.ndarray:
            X = X.to_numpy()

        random_state = check_random_state(random_state)

        check_array(X, accept_sparse = False, dtype="numeric", order=None,
                    copy=False, force_all_finite=True, ensure_2d=True,
                    allow_nd=False, ensure_min_samples = self.K,
                    ensure_min_features = self.H, estimator=None)

        X = X.astype(float)

        J = self.J
        Jlist = self.Jlist
        U = self.U
        V = self.V
        G = self.G
        weights = self.weights
        iterations = self.iterations

        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for seed in seeds:
            self.__fit_single(X, seed)
            if np.isnan(self.J):
                raise ValueError("matrix may contain negative or unexpected NaN values")
            
            if (self.J < J):
                J = self.J
                Jlist = self.Jlist
                U = self.U
                V = self.V
                G = self.G
                weights = self.weights
                iterations = self.iterations
                
        self.J = J
        self.Jlist = Jlist
        self.U = U
        self.V = V
        self.G = G
        self.weights = weights
        self.iterations = iterations

        if super().isdecreasing(self.Jlist) == False:
            warn('Objective function did not converge')

        return self


    def __prototypes_WFDK(self,X, W, Um, Vn, G):
        '''
        X is the dataset 
        KMs is the array of kernel between X and G
        Um is the fuzzy matrix of the objects into K cluster raised to the m
        Vn is the fuzzy matrix of the variables into H cluster raised to the n
        G is the current prototype matrix that will be updated
        ld is the smallest possible value for a denominator. Used to avoid indeterminacy
        '''

        ld = 10 ** (-10)

        K,H = G.shape
        G_new = np.zeros((K,H))
        P = X.shape[1]

        if np.ndim(W) == 1:
            W = np.tile(W,K).reshape((K,P))

        for k in range(K):
            uk = (Um[:,k]).reshape((-1,1))
            wk = (W[k]).reshape((1,-1))
            for h in range(H):
                vh = (Vn[:,h]).reshape((1,-1))
                wh = (uk * np.full_like(X,1) * vh * wk)
                if wh.sum() > ld:
                    G_new[k,h] = np.average(a = X, weights = wh )     
                else:
                    G_new[k,h] = G[k,h]

        return G_new



    def __get_weights_global_sum(self, D, Um, Vn, gamma):
        '''
        D is the array of distance between X and the co-closters
        Um is the fuzzy matrix of the objects into K cluster raised to the m
        Vn is the fuzzy matrix of the variables into H cluster raised to the n
        gamma is the parameter for the computation of the withgs with sum constraint
        ld is the smallest possible value for a denominator. Used to avoid indeterminacy
        '''
        ld = 10**(-100)
        K,H = D.shape[:2]
        P = D.shape[3]
        Dk = np.zeros((K,P))
        for k in range(K):
            uk = (Um[:,k]).reshape((-1,1))
            Dhp = np.zeros((H,P))
            for h in range(H):
                vh = (Vn[:,h]).reshape((1,-1))
                Dhp[h] = (uk * D[k,h] * vh).sum(axis = 0)
            Dk[k] = Dhp.sum(axis = 0)
        Dj = Dk.sum(axis = 0)
        idj = np.where(Dj > ld)[0]
        nj = len(idj)
        W = np.zeros(P)
        if nj == P:
            for j in range(P):
                ratio = (Dj[j]/Dj)**( 1/(gamma-1) )
                W[j] =  ratio.sum() ** (-1)
        else:
            Dju = Dj[idj]
            Wu = np.zeros(nj)
            for a in range(nj):
                ratio = (Dju[a]/Dju)**( 1/(gamma-1) )
                Wu[a] =  ratio.sum() ** (-1)
            W[idj] = Wu
        return W

    def __D_adaptive(self, D,W):
        '''
        D is the array of distance between X and the co-closters
        W is the weights of the variables
        '''

        if np.ndim(W) == 1:
            return D * W
        else:
            Da = np.zeros_like(D)
            K = D.shape[0]
            for k in range(K):
                Da[k] = D[k] * W[k]
            return Da



    def __fit_single(self, X, random_state = 0):  
        
        if type(X) != np.ndarray:
            X = X.to_numpy()
        X = X.astype('float64')
        N,P = X.shape

        # Inicialization step
        weights = np.full(P, (1/P))
        U = super().random_U((N,self.K),random_state+1)
        V = super().random_U((P,self.H),random_state+2)
        Um = U**self.m
        Vn = V**self.n
        G = super().initial_cocluster_prototypes(X = X, K = self.K, H = self.H)
        Wb = (weights**self.gamma)
        Jlist = []
        J = 0
        # iterative step
        t = 1
        while True:
            G = self.__prototypes_WFDK(X, Wb, Um, Vn, G)
            D = super().dist_array(X, G)
            weights = self.__get_weights_global_sum(D, Um, Vn, self.gamma)
            Wb = (weights**self.gamma)
            Da = self.__D_adaptive(D,Wb)
            U = super().getU_co_clustering(Da, U, V, self.m, self.n)
            V = super().getV_co_clustering(Da, U, V, self.m, self.n)
            Um = U**self.m
            Vn = V**self.n
            Jcurr = J
            J = super().computeJ_co_clustering(Da, Um, Vn)
            Jlist.append(J)

            if (np.abs(Jcurr - J) < self.epsilon) or t > self.T:
                break
            else:
                t = t + 1

        self.U = U
        self.V = V
        self.G = G
        self.weights = weights
        self.J = J
        self.Jlist = Jlist
        self.iterations = t

