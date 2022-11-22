from keras import backend as K
import numpy as np

class Metrics:
    @staticmethod
    def r_square(y_true, y_pred):
        
        """
        - R^2 coefficient for variance
        - The higher the R-squared, the better the model fits your data.
        """
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        return (1 - SS_res/(SS_tot + K.epsilon()))
    
    def RMSD(S1, S2):
        # Function for RMSD (Root Mean Squared Difference)
        sqdiff = (S1-S2)**2
        num = np.sum(sqdiff, axis = 1)
        den = np.sum(S1**2, axis = 1)
        msd = np.divide(num,den)
        rmsd = np.sqrt(msd)
        return rmsd
    
    def CF(S1, S2):
        # Correlation factor
        S1 = np.array(S1,dtype='float')
        S2 = np.array(S2,dtype='float')
        L = S1.shape[0]
        mu1 = np.mean(S1,axis=1)
        mu1 = mu1[:,None]
        mu2 = np.mean(S2,axis=1)
        mu2 = mu2[:,None]
        sd1 = np.std(S1,axis=1)
        sd1 = sd1[:,None]
        sd2 = np.std(S2,axis=1)
        sd2 = sd2[:,None]
        dev1 = S1 - mu1
        dev2 = S2 - mu2
        dev12 = np.multiply(dev1,dev2)
        sd12 = (1/(L-1))*np.sum(dev12,axis=1)
        sd12 = sd12[:,None]
        sd1xsd2 = np.multiply(sd1,sd2)
        C = np.divide(sd12,sd1xsd2)
        return C