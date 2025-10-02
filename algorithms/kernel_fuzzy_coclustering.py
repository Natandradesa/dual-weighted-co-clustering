import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist, squareform
#from scipy.stats.mstats import gmean
from warnings import warn
from sklearn.utils import check_array, check_random_state


#------------------------------------- Helper functions -------------------------------------#


def sigma2est(X, batch_size = 20000, random_state = 26):
    
    if type(X) != np.ndarray:
        X = X.to_numpy()

    n = np.size(X)
    X_flat = X.reshape(-1,1) # column vector
    if n <= batch_size:
        dist = pdist(X_flat, 'sqeuclidean')
        dist = dist[dist > 0.0]
        return (np.quantile(dist, [0.1, 0.9]).sum()/4)
    else:
        np.random.seed(random_state)
        np.random.shuffle(X_flat)
        num_batches = (n + batch_size - 1) // batch_size 
        rs_batches = np.zeros(num_batches)
        for i in range(num_batches):
            print(f"\rProgress: {i + 1}/{num_batches}", end="")
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n)  # Garante que o índice final não ultrapasse o limite
            X_i = X_flat[start_idx:end_idx]
            dist = pdist(X_i, 'sqeuclidean')
            dist = dist[dist > 0.0]
            rs_batches[i] = np.quantile(dist, [0.1, 0.9]).sum()/4
        return rs_batches.mean()
    
        
class BaseKernel():

    def gmean(self, x):
        return (x ** (1/len(x))).prod()

    # Helper functions for all algortihms
    def random_U(self,shape, random_state = None):
        
        if type(shape) == int:
            np.random.seed(random_state)
            h = np.random.random(shape)
            return h/h.sum()
        else:
            np.random.seed(random_state)
            h = np.random.random(shape)
            return h/h.sum(axis= 1).reshape(-1,1) 

    def initial_prototypes(self,X, Um, Vn):
        '''
        X is the dataset 
        KMs is the array of kernel between X and G
        Um is the fuzzy matrix of the objects into K cluster raised to the m
        Vn is the fuzzy matrix of the variables into H cluster raised to the n
        ld is the lowest denominator, to avoid indeterminacy.
        '''
        K, H = Um.shape[1], Vn.shape[1]
        G = np.zeros((K,H))
        for k in range(K):
            uk = (Um[:,k]).reshape((-1,1))
            for h in range(H):
                vh = (Vn[:,h]).reshape((1,-1))
                wh = (uk * np.full_like(X,1) * vh)
                G[k,h] = np.average(a = X, weights = wh )            
        return G
    

    def prototypes_adaptive(self, X, KMs, W, Um, Vn, G):
        '''
        X is the dataset 
        KMs is the array of kernel between X and G
        W is the weights of the variables
        Um is the fuzzy matrix of the objects into K cluster raised to the m
        Vn is the fuzzy matrix of the variables into H cluster raised to the n
        G is the current prototype matrix
        ld is the lowest denominator, to avoid indeterminacy.
        '''

        ld = 1e-10

        K,H = G.shape
        P = X.shape[1]
        
        if np.ndim(W) == 1:
            W = np.tile(W,K).reshape((K,P))
    
        for k in range(K):
            uk = (Um[:,k]).reshape((-1,1))
            wk = W[k].reshape((1,-1))
            for h in range(H):
                vh = (Vn[:,h]).reshape((1,-1))
                KMkh = uk * KMs[k,h]* vh * wk
                if KMkh.sum() > ld:
                    G[k,h] = np.average(a = X, weights = KMkh )            
        return G


    def gaussian_kernel_array(self, X, G, sig2):
        '''
        X is the dataset
        G is the prototypes of the co-clusters
        sig2 is the parameters of the gaussian kernel
        '''
        n, p = X.shape
        K, H = G.shape
        const = 1/(2*sig2)
        KMs = np.zeros((K, H, n, p))
        for k in range(K):
            for h in range(H):
                KMs[k,h] = np.exp( -const * (X - G[k,h]) ** 2 )     
        return KMs

    def getU(self, D, U, V, m, n):
        '''
        D is the array of the distances between the dataset and the prototype matrix
        V is the partiton of variables into H cluster
        m and n are the fuzzyness parameters
        '''
        Vn = V**n
        Vnt = np.transpose(Vn)
        vhn = Vn.sum(axis = 0)
        exponent = (1/(m-1))
        K,N = D.shape[0], D.shape[2]
        for i in range(N):
            Di = D[:,:,i,:] 
            Dvi = ((Vnt*Di).sum(axis = 2)) * (1/vhn)
            Dvi = np.where(np.isinf(Dvi),0,Dvi)
            Dvi = np.where(np.isnan(Dvi),0,Dvi)
            Dik = Dvi.sum(axis = 1)
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
                        const = 1 - sum_previous
                        # Computationally is possible that const < 0, but mathmatically const must be non-negative
                        if const >= 0:
                            U[i,idx] = const * (inv_dk_new/inv_dk_new.sum())
                        else:
                            U[i,idx] = 0
        return U 

    def getV(self, D, U, V, m, n):
        '''
        D is the array of the distances between the dataset and the prototype matrix
        V is the partiton of variables into H cluster
        m and n are the fuzzyness parameters
        '''
        Um = U**m
        Umt = np.transpose(Um)
        ukm = Um.sum(axis = 0)
        exponent = (1/(n-1))
        H,P = D.shape[1], D.shape[3]
        Dm = np.moveaxis(D,0,1)
        for j in range(P):
            Dj = Dm[:,:,:,j] 
            Duj = ((Umt*Dj).sum(axis=2)) * (1/ukm)
            Duj = np.where(np.isinf(Duj),0,Duj)
            Duj = np.where(np.isnan(Duj),0,Duj)
            Djh = Duj.sum(axis = 1)
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
                        if const >= 0:
                            V[j,idx] = const * (inv_dk_new/inv_dk_new.sum())
                        else:
                            V[j,idx] = 0.0
        return V

    def computeJ(self, D, Um, Vn):
        '''
        D is the array of the distance between the dataset and the prototype matrix
        Um is the fuzzy matrix of the objects into K cluster raised to the m
        Vn is the fuzzy matrix of the variables into H cluster raised to the n
        '''

    
        ukm = Um.sum(axis = 0)
        vhn = Vn.sum(axis = 0)

        K,H = D.shape[:2]
        Jc = 0
        for k in range(K):
            if ukm[k] != 0:
                uk = (Um[:,k]).reshape((-1,1))
                for h in range(H):
                    if vhn[h] != 0:
                        vh = (Vn[:,h]).reshape((1,-1))
                        const = (1/ukm[k]) * (1/vhn[h])
                        if ~np.isinf(const):
                            Dkh = const * ( (uk * D[k,h] * vh).sum() )
                            Jc += Dkh.sum()
        return Jc


    
    def D_adaptive(self, D,W):

        if np.ndim(W) == 1:
            return (D * W)
        else:
            Da = np.zeros_like(D)
            K = D.shape[0]
            for k in range(K):
                Da[k] = D[k] * W[k]
            return Da


class GKFDK(BaseKernel):

    def __init__(self, K, H, m, n, sigma2,  epsilon = 1e-5, max_iter = 100, n_init = 1 ):
        self.K = K
        self.H = H
        self.m = m
        self.n = n
        self.sigma2 = sigma2
        self.epsilon = epsilon
        self.T = max_iter
        self.n_init = n_init

        self.U = None
        self.V = None
        self.G = None
        self.initial_G = None
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
        initial_G = self.initial_G 
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
                initial_G = self.initial_G
                iterations = self.iterations
                
        self.J = J
        self.Jlist = Jlist
        self.U = U
        self.V = V
        self.G = G
        self.initial_G = initial_G
        self.iterations = iterations

        return self

    def __prototypes(self, X, KMs, Um, Vn, G):
        '''
        X is the dataset 
        KMs is the array of kernel between X and G
        W is the weights of the variables
        Um is the fuzzy matrix of the objects into K cluster raised to the m
        Vn is the fuzzy matrix of the variables into H cluster raised to the n
        G is the current prototype matrix
        ld is the lowest denominator, to avoid indeterminacy.
        '''
        ld = 1e-10
        K,H = G.shape    
        for k in range(K):
            uk = (Um[:,k]).reshape((-1,1))
            for h in range(H):
                vh = (Vn[:,h]).reshape((1,-1))
                KMkh = uk * KMs[k,h]* vh 
                if KMkh.sum() > ld:
                    G[k,h] = np.average(a = X, weights = KMkh )            
        return G

    def __fit_single(self, X, random_state = 0):  
        
        if type(X) != np.ndarray:
            X = X.to_numpy()
        X = X.astype('float64')
        N,P = X.shape


        # Inicialization step
        U = super().random_U((N,self.K),random_state + 1)
        V = super().random_U((P,self.H),random_state + 2)
        Uinit = U.copy()
        Vinit = V.copy()
        Um = U**self.m
        Vn = V**self.n
        G = super().initial_prototypes(X, Um, Vn)
        initial_G = G.copy()
        KM = super().gaussian_kernel_array(X, G, self.sigma2)
        D = (2-2*KM)
        J = 0
        Jlist = []

        # iterative step
        t = 1
        while True:
            G = self.__prototypes(X, KM, Um, Vn, G)
            KM = super().gaussian_kernel_array(X, G, self.sigma2)
            D = (2 - 2*KM)
            U = super().getU(D, U, V, self.m, self.n)
            V = super().getV(D, U, V, self.m, self.n)
            Um = U**self.m
            Vn = V**self.n
            Jcurr = J
            J = super().computeJ(D, Um, Vn)
            Jlist.append(J)

            if np.abs(Jcurr - J) < self.epsilon or t > self.T:
                break
            else:
                t = t + 1

        #The following part of the code checks if there have been any technical problems with the 
        # values of m and n. If these values are too small, so that 1/m-1 or 1/n-1 are computationally 
        # infinite. The matrices U and V are not updated. Therefore, these values are not appropriate

        if (Uinit == U).all() and (Vinit == V).all():
            print("U and V didn't change")
            chg = False
        elif (Uinit == U).all():
            print("U didn't change")
            chg = False
        elif (Vinit == V).all():
            print("V didn't change")
            chg = False
        else:
            #print("all right")
            chg = True

        self.initial_G = initial_G
        self.U = U
        self.V = V
        self.G = G
        self.J = J
        self.Jlist = Jlist
        self.iterations = t

        return self


class WGKFDK(BaseKernel):

    def __init__(self, K, H, m, n, sigma2,  epsilon = 1e-5, max_iter = 100, n_init = 1 ):
        self.K = K
        self.H = H
        self.m = m
        self.n = n
        self.sigma2 = sigma2
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

        #if super().isdecreasing(self.Jlist) == False:
        #    warn('Objective function did not converge')

        return self


    def __get_weights_local_prod(self,D, Um, Vn, W):
        '''
        D is the array of distance between X and the co-closters
        Um is the fuzzy matrix of the objects into K cluster raised to the m
        Vn is the fuzzy matrix of the variables into H cluster raised to the n
        W is the weights of the variables at previous iterations
        ld is the lowest denominator, to avoid indeterminacy.
        '''
        ld = 10**(-100)
        vhn = Vn.sum(axis = 0)
        K,H,N,P = D.shape
        for k in range(K):
            uik = (Um[:,k]).reshape((-1,1))
            DN = np.zeros((H,P))
            for h in range(H):
                vjh = (Vn[:,h])
                DN[h] = (uik * D[k,h]).sum(axis = 0) * (vjh/vhn[h])
            DN = np.where(np.isinf(DN),0,DN)
            DHN = DN.sum(axis = 0)
            jless = np.where(DHN<= ld)[0]
            nless = len(jless)
            if nless == 0:
                W[k] =  super().gmean(DHN)/DHN
            else:
                jupper = np.where(DHN > ld)[0]
                den = np.prod(W[k][jless])
                if den > ld and nless < P:
                    const = (1/den)**(1/(P-nless))
                    if  ~np.isnan(const) and ~np.isinf(const):
                        sum_upper = DHN[jupper]
                        W[k][jupper] = (const*super().gmean(sum_upper)/sum_upper)
    
        return W


    def __fit_single(self, X, random_state = 0):  
        
        if type(X) != np.ndarray:
            X = X.to_numpy()
        X = X.astype('float64')
        N,P = X.shape


        # Inicialization step
        U = super().random_U((N,self.K),random_state + 1)
        V = super().random_U((P,self.H),random_state + 2)
        Uinit = U.copy()
        Vinit = V.copy()
        Um = U**self.m
        Vn = V**self.n
        G = super().initial_prototypes(X, Um, Vn)
        KM = super().gaussian_kernel_array(X, G, self.sigma2)
        weights = np.ones((self.K,P))
        D = (2-2*KM)
        J = 0
        Jlist = []

        # iterative step
        t = 1
        while True:
            G = super().prototypes_adaptive(X, KM, weights, Um, Vn, G)
            KM = super().gaussian_kernel_array(X, G, self.sigma2)
            D = (2 - 2*KM)
            weights = self.__get_weights_local_prod(D, Um, Vn, weights)
            Da = super().D_adaptive(D, weights)
            U = super().getU(Da, U, V, self.m, self.n)
            V = super().getV(D, U, V, self.m, self.n)
            Um = U**self.m
            Vn = V**self.n
            Jcurr = J
            J = super().computeJ(Da, Um, Vn)
            Jlist.append(J)

            if np.abs(Jcurr - J) < self.epsilon or t > self.T:
                break
            else:
                t = t + 1

        #The following part of the code checks if there have been any technical problems with the 
        # values of m and n. If these values are too small, so that 1/m-1 or 1/n-1 are computationally 
        # infinite. The matrices U and V are not updated. Therefore, these values are not appropriate.s

        if (Uinit == U).all() and (Vinit == V).all():
            print("U and V didn't change")
            chg = False
        elif (Uinit == U).all():
            print("U didn't change")
            chg = False
        elif (Vinit == V).all():
            print("V didn't change")
            chg = False
        else:
            #print("all right")
            chg = True


        self.U = U
        self.V = V
        self.G = G
        self.weights = weights
        self.J = J
        self.Jlist = Jlist
        self.iterations = t

        return self

