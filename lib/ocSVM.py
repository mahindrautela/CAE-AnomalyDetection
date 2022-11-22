from sklearn import svm
import numpy as np

class ocSVM:
    @staticmethod
    def svmbuild(traindata,n):
        model = svm.OneClassSVM(kernel="rbf", nu = n)
        model.fit(traindata)
        
        return model