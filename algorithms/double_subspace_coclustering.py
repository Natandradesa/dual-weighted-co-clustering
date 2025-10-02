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

    def compute_final_weights(self, distance_metric, weights, lowest_denominator):
        l = len(weights)
        D = distance_metric
        ld = lowest_denominator
        if (D > ld).all():
            return self.gmean(D)/D
        else:
            id_less = np.where(D <= ld)[0]
            id_upper = np.where(D > ld)[0]
            nless = len(id_less)
            if nless < l:
                const = (1/np.prod(weights[id_less]))**(1/(l-nless))
                if const > ld and  ~np.isnan(const) and ~np.isinf(const):
                    sum_upper = D[id_upper]
                    weights[id_upper] = (const * self.gmean(sum_upper)/sum_upper)
        return weights


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
        exponent = (1/(m-1))
        K,N = D.shape[0], D.shape[2]
        for i in range(N):
            Di = D[:,:,i,:] 
            Dvi = ((Vnt*Di).sum(axis = 2)) 
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
        exponent = (1/(n-1))
        H,P = D.shape[1], D.shape[3]
        Dm = np.moveaxis(D,0,1)
        for j in range(P):
            Dj = Dm[:,:,:,j] 
            Duj = ((Umt*Dj).sum(axis=2))
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

        K,H = D.shape[:2]
        Jc = 0
        for k in range(K):
            uk = (Um[:,k]).reshape((-1,1))
            for h in range(H):
                vh = (Vn[:,h]).reshape((1,-1))
                Dkh = (uk * D[k,h] * vh).sum() 
                Jc += Dkh.sum()
        return Jc
    
    def D_adaptive(self, D, object_weights, variable_weights):

        if np.ndim(object_weights) == 1 and np.ndim(variable_weights) == 1 :
            ow = object_weights.reshape(-1,1)
            vw = variable_weights.reshape(1,-1)
            return (ow * D * vw)
        else:
            Da = np.zeros_like(D)
            K,H,_,_ = D.shape
            for k in range(K):
                vw = variable_weights[k].reshape(1,-1)
                for h in range(H):
                    ow = object_weights[h].reshape(-1,1)
                    Da[k,h] = ow * D[k,h] * vw
            return Da

    def isdecreasing(self, J):
        J_sorted = J.copy()
        J_sorted.sort(reverse = True)
        if J_sorted == J:
            return True
        else:
            return False

class DWGKFDK(BaseKernel):

    def __init__(self, K, H, m, n, sigma2,  epsilon = 1e-5, max_iter = 100, n_init = 1, lowest_denominator = 1e-10 ):

        """
        Implementation of the algorithm Dual Weighted Gaussian Kernel Fuzzy Double K-means (DWGKFDK)

        Parameters
        ----------
        K : int
            Number of object clusters (K >= 2).
        H : int
            Number of variable clusters (H >= 2).
        m : float
            Fuzzification degree of the objects (m > 1).
        n : float
            Fuzzification degree of the variables (n > 1)
        sigma2 : float
            Width parameter of the Gaussian kernel
        epsilon : float
            Stopping criterion (tolerance).
        max_iter : int
            Maximum number of iterations.
        n_init : int
            Number of different initializations to test.
        lowest_denominator : float
            Minimum value to avoid division by zero.

        Attributes
        ----------
        initial_U : ndarray of shape (n, K)
            Initial fuzzy partition matrix of the objects.
        initial_V : ndarray of shape (K, p)
            Initial fuzzy partition matrix of the variables.
        U : ndarray of shape (n, K)
            Final fuzzy partition matrix of the objects.
        V : ndarray of shape (K, p)
            Final fuzzy partition matrix of the variables.
        G : ndarray of shape (K, H)
            Prototype matrix.
        object_weights : ndarray
            Weights assigned to objects.
        variable_weights : ndarray
            Weights assigned to variables.
        J : float
            Final value of the objective function.
        Jlist : list of float
            History of objective function values over iterations.
        iterations : int
            Number of iterations performed untill convergence.

        Methods
        -------
        fit(X)
            Runs the algorithm on dataset X.
        """




        self.K = K
        self.H = H
        self.m = m
        self.n = n
        self.sigma2 = sigma2
        self.epsilon = epsilon
        self.T = max_iter
        self.n_init = n_init
        self.ld = lowest_denominator

        self.initial_U = None
        self.initial_V = None
        self.U = None
        self.V = None
        self.G = None
        self.object_weights = None
        self.variable_weights = None
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

        initial_U = self.initial_U
        initial_V = self.initial_V
        U = self.U
        V = self.V
        G = self.G
        object_weights = self.object_weights
        variable_weights = self.variable_weights
        J = self.J
        Jlist = self.Jlist
        iterations = self.iterations

        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for seed in seeds:
            self.__fit_single(X, seed)
            if np.isnan(self.J):
                raise ValueError("matrix may contain negative or unexpected NaN values")
            
            if (self.J < J):
                initial_U = self.initial_U
                initial_V = self.initial_V
                U = self.U
                V = self.V
                G = self.G
                object_weights = self.object_weights
                variable_weights = self.variable_weights
                J = self.J
                Jlist = self.Jlist
                iterations = self.iterations
                
        self.initial_U = initial_U
        self.initial_V = initial_V
        self.U = U
        self.V = V
        self.G = G
        self.object_weights = object_weights
        self.variable_weights = variable_weights
        self.J = J
        self.Jlist = Jlist
        self.iterations = iterations

        if super().isdecreasing(self.Jlist) == False:
            warn('Objective function did not converge')

        return self


    def __prototypes(self, X, KMs, object_weights, variable_weights, Um, Vn, G):
        K,H = G.shape
        ow = object_weights.reshape(-1,1)
        vw = variable_weights.reshape(1,-1)
  
        for k in range(K):
            uk = (Um[:,k]).reshape((-1,1))
            for h in range(H):
                vh = (Vn[:,h]).reshape((1,-1))
                KMkh = uk * ow * KMs[k,h]* vh * vw
                if KMkh.sum() > self.ld:
                    G[k,h] = np.average(a = X, weights = KMkh )            
        return G
    
    def __get_weights(self,D, Um, Vn, object_weights, variable_weights):

        vw = variable_weights.reshape(1,-1)
        #vhn = Vn.sum(axis = 0)
        #ukm = Um.sum(axis = 0)
        K,H,N,P = D.shape
        Dkh = np.zeros((N,P))
        for k in range(K):
            uik = (Um[:,k]).reshape((-1,1))
            for h in range(H):
                vjh = (Vn[:,h]).reshape((1,-1))
                Dkh += (uik * D[k,h] * vjh) 

        # computing the weights of the objects
        Di = (Dkh * vw).sum(axis = 1)
        object_weights = super().compute_final_weights(distance_metric=Di, weights=object_weights,
                                                         lowest_denominator=self.ld)

        ow = object_weights.reshape(-1,1)

        # computing the weights of the variables
        Dj = (Dkh * ow).sum(axis = 0)
        variable_weights = super().compute_final_weights(distance_metric=Dj, weights=variable_weights,
                                                         lowest_denominator=self.ld)
        
        return object_weights, variable_weights



    def __fit_single(self, X, random_state = 0):  
        
        if type(X) != np.ndarray:
            X = X.to_numpy()
        X = X.astype('float64')
        N,P = X.shape


        # Inicialization step
        U = super().random_U((N,self.K),random_state + 1)
        V = super().random_U((P,self.H),random_state + 2)
        self.initial_U, self.initial_V = U, V
       
        Um = U ** self.m
        Vn = V ** self.n
        G = super().initial_prototypes(X, Um, Vn)
        KM = super().gaussian_kernel_array(X, G, self.sigma2)
        object_weights, variable_weights = np.ones(N), np.ones(P)
        
        D = (2.0-2.0*KM)
        J = super().computeJ(D = D, Um = Um, Vn = Vn)
        Jlist = [J]

        # iterative step
        t = 0
        while True:
            G = self.__prototypes(X=X, KMs=KM, object_weights=object_weights, variable_weights=variable_weights,
                                  Um=Um, Vn=Vn, G=G)
            KM = super().gaussian_kernel_array(X=X, G=G, sig2=self.sigma2)
            D = (2.0 - 2.0*KM)
            object_weights, variable_weights = self.__get_weights(D=D, Um=Um, Vn=Vn, object_weights=object_weights,
                                                                  variable_weights=variable_weights)
            Da = super().D_adaptive(D=D, object_weights=object_weights, variable_weights=variable_weights)

            U = super().getU(D=Da, U=U, V=V, m=self.m, n=self.n)
            V = super().getV(D=Da, U=U, V=V, m=self.m, n=self.n)
    
            Um = U ** self.m
            Vn = V ** self.n
            
            Jcurr = J
            J = super().computeJ(D=Da, Um=Um, Vn=Vn)
            Jlist.append(J)

            if np.abs(Jcurr - J) < self.epsilon or t > self.T:
                break
            else:
                t = t + 1

        self.U = U
        self.V = V
        self.G = G
        self.object_weights = object_weights
        self.variable_weights = variable_weights
        self.J = J
        self.Jlist = Jlist
        self.iterations = t+1


        return self
