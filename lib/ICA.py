import numpy as np
from sklearn.decomposition import FastICA  

class ICAmodel:
    @staticmethod
    def icabuild(data,dim):
        nd = dim
        ica = FastICA(n_components=nd, random_state = np.random.RandomState(42))
        ica.fit(data)
        trans_ica = ica.transform(data)
        
        explained_var = np.var(trans_ica, axis=0)
        explained_var_ratio = explained_var / np.sum(explained_var)
        img_reduced = ica.inverse_transform(trans_ica)
        
        return (trans_ica, explained_var_ratio, img_reduced)