import pandas as pd
import numpy as np
import cv2
import os

class ImportImgData3:
    @staticmethod
    # import dataset
    # import dataset
    def load_labels(m):
        dfUD = pd.DataFrame(np.zeros((m,1), dtype=int))
        dfD = pd.DataFrame(np.ones((m,1), dtype=int))
        return dfUD, dfD
    
    def load_imagesUD(dfUD,pathUD):
        imagesUD = []
        for i in dfUD.index.values:
            baseUD = os.path.sep.join([pathUD, "base{}.png".format(i + 1)])
            image = cv2.imread(baseUD) # read the path using opencv
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # cv2.COLOR_BGR2GRAY
            image = cv2.resize(image, (256, 256))
            #plt.imshow(image) # use matplotlib to plot the image
            #image = image[:,:,np.newaxis] #This is convert (600,600) --> (600,600,1)
            imagesUD.append(image) 
        return np.array(imagesUD)
    
    def load_imagesD(dfD,pathD):
        imagesD = []
        for i in dfD.index.values:
            baseD = os.path.sep.join([pathD, "dam{}.png".format(i + 1)])
            image = cv2.imread(baseD) # read the path using opencv
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))
            #plt.imshow(image) # use matplotlib to plot the image
            #image = image[:,:,np.newaxis] #This is convert (600,600) --> (600,600,1)
            imagesD.append(image) 
        return np.array(imagesD)