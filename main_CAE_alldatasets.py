"""
Convolutional autoencoders for anomaly detection
"""
import lib.tf_silent
import numpy as np
import cv2
import os
import time
import tensorflow as tf
from lib.networkAE import Network
import matplotlib.pyplot as plt
from lib.visualizationAE import Visualize
from lib.importdatasetsOGW import ImportImgData1
from lib.importdatasetsNASA import ImportImgData2
from lib.importdatasetsUoNCT import ImportImgData3

from lib.performancemetrics import Metrics
from lib.visualizationAE import Visualize
from tensorflow.keras.optimizers import Adam

# tensorflow version
print("Tensorflow version = ",tf.__version__)

# calculate the reconstruction error
def reconstruction_error(samples, pred_samples):
    errors = []
    for (image, recon) in zip(samples, pred_samples):
        mse = np.mean((image - recon)**2)
        errors.append(mse)
    return errors

############################################################################
############################ Import datasets ###############################
############################################################################

choosedataset = 2
print("The dataset selected (1 = OGW, 2 = NASA, 3 = UoNCT) ===>>>>",
      choosedataset)

# finding the preference vector for the choosen dataset
if choosedataset == 1: 
    Les = 2.5e-7
    VLes = 2.5e-7
    split = 0.15
    nepochs = 1500
    lr = 1e-3
    batchsize = 32
    imgidx = 1300
elif choosedataset == 2:
    Les = 1e-6
    VLes = 1e-6
    split = 0.15
    nepochs = 3500
    lr = 1e-4
    batchsize = 20
    imgidx = 150
elif choosedataset == 3:
    Les = 1e-7
    VLes = 1e-7
    split = 0.064
    nepochs = 1500
    lr = 1e-3
    batchsize = 16
    imgidx = 1300
    
# preference vector
prefvect = [Les,VLes,split,nepochs,lr,batchsize,imgidx]

print("The preference vector for dataset" + str(choosedataset) 
      + "==>", prefvect)

# callbacks for early stopping
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('val_loss')<VLes) and (logs.get('loss')<Les):
      print("\nReached perfect loss so cancelling training!")
      self.model.stop_training = True

epoch_schedule = myCallback()

# size of the latent space
nd = 3
print("Size of latent space ===>>>", nd)

############################## DATSETS #####################################

if choosedataset == 1:
    # Import Dataset  - OGW
    # dataset name
    datasetname = "OGW"
    datasetidx = "1"
    
    # Labels
    [dfUD,dfD] = ImportImgData1.load_labels(18000)
    df = np.concatenate([dfUD, dfD], axis=0)
    print("Shape of the Labels",df.shape)
    
    # location of images
    pathD = "E:/OneDrive - Indian Institute of Science/PhD-MSR/OGWdataset/"\
              "Wavelet_Dataset/Wavelet_dataset-1/Damage"
    pathUD = "E:/OneDrive - Indian Institute of Science/PhD-MSR/OGWdataset/"\
              "Wavelet_Dataset/Wavelet_dataset-1/Baseline"
    
    # pathUD = "E:/OneDrive - Indian Institute of Science/PhD-MSR/OGWdataset/"\
    #         "New_CWTdataset/Baseline"
            
    # pathD = "E:/OneDrive - Indian Institute of Science/PhD-MSR/OGWdataset/"\
    # "New_CWTdataset/Damage"
    
    # Import the D images
    imagesUD = ImportImgData1.load_imagesUD(dfUD, pathUD)
    imagesUD = imagesUD.astype('float32')
    imagesUD = imagesUD / 255
    
    # Import the D images
    imagesD = ImportImgData1.load_imagesD(dfD, pathD)
    imagesD = imagesD.astype('float32')
    imagesD = imagesD / 255
    
    print("Shape of Healthy images", imagesUD.shape)
    print("Shape of UnHealthy images",imagesD.shape)
    
    # Select 5000 examples randomly from dataset
    mex = 2200
    n1 = np.random.randint(0, 18000, mex)
    imUD1 = imagesUD[n1]
    imD1 = imagesD[n1]
    print("New Undamaged dataset",imUD1.shape)
    print("New damaged dataset",imD1.shape)
    
    # delete the previous arrays to save space
    del imagesUD
    del imagesD
    
    # Test the importing 
    images = []
    for i in range(2):
        base = os.path.sep.join([pathUD, "Baseline-{}.png".format(i + 1)])
        print("Location of the image", base)
        image0 = cv2.imread(base) # read the path using opencv
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB) # BGR2GRAY for B&W
        image0 = cv2.resize(image0, (256, 256))
        plt.title("Image from the dataset-"+ datasetidx)
        plt.imshow(image0) # use matplotlib to plot the image
        #image = image[:,:,np.newaxis] # convert (600,600) --> (600,600,1)
        images.append(image0)


elif choosedataset == 2:
    # Import Dataset  - NASA 
    # dataset name
    datasetname = "NASA"
    datasetidx = "2"
    
    # Labels
    [dfUD,dfD] = ImportImgData2.load_labels(252)
    df = np.concatenate([dfUD, dfD], axis=0)
    print("Shape of the Labels",df.shape)
    
    # location of images
    mainpath = "E:/OneDrive - Indian Institute of Science/PhD-MSR/NASAdataset/" \
               "Layup2/L2S11/CWTdataset_noise/"
    folder = ['UD','UDn1','UDn2','UDn3','D1k', 'D10k','D20k']
    
    pathUD1 = mainpath + folder[0]
    pathUD2 = mainpath + folder[1]
    pathUD3 = mainpath + folder[2]
    pathUD4 = mainpath + folder[3]
    pathD1 = mainpath + folder[4]
    pathD2 = mainpath + folder[5]
    pathD3 = mainpath + folder[6]
    
    # Import the UD images
    imUD1 = ImportImgData2.load_imagesUD(dfUD, pathUD1, 1)
    imUD1 = imUD1.astype('float32')
    imUD1 = imUD1 / 255
    
    # Import the UD images
    imUD2 = ImportImgData2.load_imagesUD(dfUD, pathUD2, 2)
    imUD2 = imUD2.astype('float32')
    imUD2 = imUD2 / 255
    
    # Import the UD images
    imUD3 = ImportImgData2.load_imagesUD(dfUD, pathUD3, 3)
    imUD3 = imUD3.astype('float32')
    imUD3 = imUD3 / 255
    
    # Import the UD images
    imUD4 = ImportImgData2.load_imagesUD(dfUD, pathUD4, 4)
    imUD4 = imUD4.astype('float32')
    imUD4 = imUD4 / 255
    
    # Import the D images
    imD1 = ImportImgData2.load_imagesD(dfD, pathD1)
    imD1 = imD1.astype('float32')
    imD1 = imD1 / 255
    
    # Import the D images
    imD2 = ImportImgData2.load_imagesD(dfD, pathD2)
    imD2 = imD2.astype('float32')
    imD2 = imD2 / 255
    
    # Import the D images
    imD3 = ImportImgData2.load_imagesD(dfD, pathD3)
    imD3 = imD3.astype('float32')
    imD3 = imD3 / 255
    
    print("Shape of Healthy images",imUD1.shape)
    print("Shape of UnHealthy images",imD1.shape)
    
    # Test the UD importing 
    i = 0
    base1 = os.path.sep.join([pathUD1, "UD_{}.png".format(i + 1)])
    img1 = cv2.imread(base1) # read the path using opencv 
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (256, 256))
    plt.title("Image from the dataset-" + datasetidx)
    plt.imshow(img1)
    print(img1.shape) 
        
    # test the D importing
    base2 = os.path.sep.join([pathD1, "D_{}.png".format(i + 1)])
    img2 = cv2.imread(base2) # read the path using opencv
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(img2, (256, 256))
    plt.title("Image from the dataset-" + datasetidx)
    plt.imshow(img2)
    print(img2.shape)


elif choosedataset == 3:
    # Import-dataset UoNCT 
    # dataset name
    datasetname = "UoNCT"
    datasetidx = "3"
    
    # Labels
    [dfUD,dfD] = ImportImgData3.load_labels(1560)
    df = np.concatenate([dfUD, dfD], axis=0)
    print("Shape of the Labels",df.shape)
    
    # location of images
    pathD = "E:/OneDrive - Indian Institute of Science/PhD-MSR/UoNCTdataset/"\
             "Experimental_data/DATA/Dataset/wavelet/dam"
    pathUD = "E:/OneDrive - Indian Institute of Science/PhD-MSR/UoNCTdataset/"\
             "Experimental_data/DATA/Dataset/wavelet/base"
    
    # Import the UD images
    imUD1 = ImportImgData3.load_imagesUD(dfUD, pathUD)
    imUD1 = imUD1.astype('float32')
    imUD1 = imUD1 / 255
    
    # Import the D images
    imD1 = ImportImgData3.load_imagesD(dfD, pathD)
    imD1 = imD1.astype('float32')
    imD1 = imD1 / 255
    
    print("Shape of Healthy images", imUD1.shape)
    print("Shape of UnHealthy images",imD1.shape)
    
    # Test the importing 
    images = []
    for i in range(2):
        #print(i)
        base = os.path.sep.join([pathUD, "base{}.png".format(i + 1)])
        #print(base)
        image0 = cv2.imread(base) # read the path using opencv 
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB) # BGR2GRAY for B&W
        image0 = cv2.resize(image0, (256, 256))
        plt.title("Image from the dataset-" + datasetidx)
        plt.imshow(image0) # use matplotlib to plot the image
        #image = image[:,:,np.newaxis] # Convert (600,600) --> (600,600,1)
        images.append(image0)


############################### Train - Test splitting ######################   
# construct the training and testing split for baseline images
from sklearn.model_selection import train_test_split
(trainX, testX) = train_test_split(imUD1, test_size = split, random_state=42)
print("Shape of training set", trainX.shape)
print("Shape of validation set", testX.shape)

###############################    Training CAE  ############################ 
# Construct the model
(encoder, decoder, autoencoder) = Network.build(256, 256, 3, 
                                                filters = (16, 32, 64, 128, 256),
                                                neurons = 100, latentDim = nd)

encoder.summary()
decoder.summary()


# Optimizer
EPOCHS = nepochs
INIT_LR = lr
BS = batchsize
opt = Adam(learning_rate=INIT_LR)
autoencoder.compile(loss="mse", 
                    optimizer=opt, 
                    metrics = ['mae',Metrics.r_square])

# Train the autoencoder
begin = time.time()

H = autoencoder.fit(
    trainX, trainX,
    validation_data=(testX, testX),
    epochs=EPOCHS,
    batch_size = BS, verbose = 2,
    callbacks = [epoch_schedule])

end = time.time()
totaltime = end-begin
print("Total runtime for CAE on dataset" + datasetidx + "==>>",totaltime)

# Visualize the training
Visualize.viz_MSEloss(H,[0,nepochs,0,1e-3],datasetidx)
Visualize.viz_MAE(H,[0,nepochs,0,5e-2],datasetidx)
Visualize.viz_R2(H,[0,nepochs,0.99,1],datasetidx)

# Visualize the reconstruction
trainXpred = autoencoder.predict(trainX)
print("Shape of prediction dataset", trainXpred.shape)

import seaborn as sns
sns.set(color_codes=True)
imgidx = 1002

# Reconstructed image
fig = plt.figure(figsize = (10, 7.2)) 
fig.add_subplot(121)
plt.title("Original Image",fontsize=15)
plt.imshow(trainX[imgidx])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.add_subplot(122)
plt.title("Reconstructed Image (CAE)",fontsize=15)
plt.imshow(trainXpred[imgidx])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("recon_cae"+datasetidx+".png", dpi=600)
plt.show()

# reconstruction error
original = trainX[imgidx]
recon = trainXpred[imgidx]
error_img = reconstruction_error(original, recon)
print("Image index of reconstruction = ", imgidx)
print("MSE of the reconstruction", np.mean(error_img))


#############################################################################
##############         Anomaly Detection using AE          ##################
#############################################################################



# 1. Plot the reconstruction errors
errors_train = reconstruction_error(trainX, trainXpred)
print("Shape of reconstruction error on trained training set", 
      np.array(errors_train).shape)
plt.figure(figsize=(20,6))
plt.hist(errors_train, bins = 50)
plt.xlabel("Reconstruction error",fontsize=22)
plt.ylabel("No of samples",fontsize=22)
plt.show()

# 1b. Basic stastistics of the reconstruction error
print("Mean of reconstruction error (CAE)",np.mean(errors_train))
print("SD of reconstruction error (CAE)",np.std(errors_train))
print("Median of reconstruction error (CAE)",np.median(errors_train))

# 2. Threshold setting for Anomaly detection
"""
1. compute the q-th quantile of the errors which serves as our
threshold to identify anomalies -- any data point that our model
reconstructed with > threshold error will be marked as an outlier.

2. Get reconstruction loss threshold based on maximum value
""" 
threshold1 = np.quantile(errors_train, 0.99)
print("Threshold 1: {}".format(threshold1))
threshold2 = np.max(errors_train)
print("Threshold 2: ", threshold2)

# 3. Prediction on validation "healthy" samples
testX_ud = testX
testXpred = autoencoder.predict(testX_ud)
print("Shape of prediction healthy samples",testXpred.shape)

errors_ud = reconstruction_error(testX_ud, testXpred)
print("Max.error in ud set",np.max(errors_ud))
print("Min.error in ud set",np.min(errors_ud))
plt.figure(figsize=(20,6))
plt.hist(errors_ud, bins = 156)
plt.xlabel("Reconstruction error on UD samples",fontsize=22)
plt.ylabel("No of samples",fontsize=22)
plt.show()

anomalies = errors_ud > threshold2
print("Number of anomaly samples in the healthy dataset: ", np.sum(anomalies))
print("Indices of anomaly samples in the healthy dataset: ", np.where(anomalies))

# 4. Prediction on validation "unhealthy" samples
testX_d = imD1
print("Shape of prediction healthy samples",testX_d.shape)

testXpred_d = autoencoder.predict(testX_d)
print(testXpred_d.shape)

errors_d = reconstruction_error(testX_d, testXpred_d)
print("Max.error in d set",np.max(errors_d))
print("Min.error in d set",np.min(errors_d))
errors_all = np.concatenate((errors_ud, errors_d),axis = 0)
sd_errors = np.std(errors_all)

plt.figure(figsize=(20,6))
plt.hist(errors_d, bins = 156)
plt.xlabel("Reconstruction error on D samples",fontsize=22)
plt.ylabel("No of samples",fontsize=22)
plt.show()

anomalies_d = errors_d > threshold2
print("Number of anomaly samples in the damaged dataset: ", np.sum(anomalies_d))
print("Indices of anomaly samples in the damaged dataset: ", np.where(anomalies_d))

# 5. Plots
Visualize.viz_thresh_ud(errors_ud,threshold1,threshold2,datasetidx)
Visualize.viz_thresh_d(errors_ud,errors_d,threshold1,threshold2,datasetidx)
Visualize.viz_thresh_all(errors_ud,errors_d,threshold1,threshold2,datasetidx)

# 6. Plot latent space
Visualize.viz_latspc(encoder,trainX,testX_ud,testX_d,[1,2])
Visualize.viz_latspc3D(encoder,trainX,testX_ud,testX_d,datasetidx)