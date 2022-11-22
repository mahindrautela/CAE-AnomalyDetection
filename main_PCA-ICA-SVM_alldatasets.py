""" 
PCA and ICA for dimesionality reduction on images + 
One class-SVM for feature learning for anomaly detection.
"""
import lib.tf_silent
import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from lib.visualizationAE import Visualize
from lib.importdatasetsOGW import ImportImgData1
from lib.importdatasetsNASA import ImportImgData2
from lib.importdatasetsUoNCT import ImportImgData3
from sklearn.model_selection import train_test_split
from lib.PCA import PCAmodel
from lib.ICA import ICAmodel
from lib.ocSVM import ocSVM
import time
print("Tensorflow version => ",tf.__version__)

# calculate the reconstruction error
def reconstruction_error(samples, pred_samples):
    errors = []
    for (image, recon) in zip(samples, pred_samples):
        mse = np.mean((image - recon)**2)
        errors.append(mse)
    return errors

def viz_latspc3D(trainX,testX_ud,testX_d,name,i):
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    plt.title('Latent space for Dataset-' + i,fontsize=22)
    ax.scatter3D(trainX[:,0],trainX[:,1],trainX[:,2],
                c='green',marker="^")
    ax.scatter3D(testX_ud[:,0],testX_ud[:,1],testX_ud[:,2],
                c='green',marker="o")
    ax.scatter3D(testX_d[:,0],testX_d[:,1],testX_d[:,2],
                c='red',marker="o")
    plt.legend(['Baseline-Train', 'Baseline-Test','Delaminated-Test'], 
               loc='best',fontsize=18)
    ax.view_init(-160, 60) 
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='z', labelsize=20)
    plt.savefig(name + i + ".png", dpi=600)
    plt.show()
        
############################################################################
############################ Import datasets ###############################
############################################################################

choosedataset = 1
print("The dataset selected (1 = OGW, 2 = NASA, 3 = UoNCT) ===>>>>",
      choosedataset)
# Reconstructed image
if choosedataset == 3: 
    idx = 1300
elif choosedataset == 1:
    idx = 1300
elif choosedataset == 2:
    idx = 150
    
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
        print(i)
        base = os.path.sep.join([pathUD, "Baseline-{}.png".format(i + 1)])
        print(base)
        image0 = cv2.imread(base) # read the path using opencv #,cv2.IMREAD_GRAYSCALE
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        image0 = cv2.resize(image0, (256, 256))
        plt.imshow(image0) # use matplotlib to plot the image
        #image = image[:,:,np.newaxis] #This is convert (600,600) --> (600,600,1)
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
    mainpath = "D:/OneDrive - Indian Institute of Science/PhD-MSR/NASAdataset/" \
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
    img1 = cv2.imread(base1) # read the path using opencv #,cv2.IMREAD_GRAYSCALE
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2.COLOR_BGR2GRAY #cv2.COLOR_BGR2RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (256, 256))
    plt.imshow(img1)
    print(img1.shape) 
        
    # test the D importing
    base2 = os.path.sep.join([pathD1, "D_{}.png".format(i + 1)])
    img2 = cv2.imread(base2) # read the path using opencv #,cv2.IMREAD_GRAYSCALE
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2.COLOR_BGR2GRAY #cv2.COLOR_BGR2RGB
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(img2, (256, 256))
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
    imagesUD = ImportImgData3.load_imagesUD(dfUD, pathUD)
    imagesUD = imagesUD.astype('float32')
    imagesUD = imagesUD / 255
    
    # Import the D images
    imagesD = ImportImgData3.load_imagesD(dfD, pathD)
    imagesD = imagesD.astype('float32')
    imagesD = imagesD / 255
    
    print("Shape of Healthy images", imagesUD.shape)
    print("Shape of UnHealthy images",imagesD.shape)
    
    # Test the importing 
    images = []
    for i in range(2):
        #print(i)
        base = os.path.sep.join([pathUD, "base{}.png".format(i + 1)])
        #print(base)
        image0 = cv2.imread(base) # read the path using opencv #,cv2.IMREAD_GRAYSCALE
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        image0 = cv2.resize(image0, (256, 256))
        plt.imshow(image0) # use matplotlib to plot the image
        #image = image[:,:,np.newaxis] #This is convert (600,600) --> (600,600,1)
        images.append(image0)
    
    imUD1 = imagesUD
    imD1 = imagesD

################################ flatenning dataset #########################

imUD1_flat = imUD1.reshape(imUD1.shape[0],
                           imUD1.shape[1]*imUD1.shape[2]*imUD1.shape[3])

imD1_flat = imD1.reshape(imD1.shape[0],
                         imD1.shape[1]*imD1.shape[2]*imUD1.shape[3])

##############################################################################
######################   Dimesionality reduction   ###########################
##############################################################################

############# ICA
print("=========>>>>>> ICA for dimensionality reduction.... ")
ica = ICAmodel()

(Xtr_ica, evr_tr_ica, recon_tr_ica) = ica.icabuild(imUD1_flat, nd)
(Xte_ica, evr_te_ica, recon_te_ica) = ica.icabuild(imD1_flat, nd)

imgrecon_tr_ica = recon_tr_ica.reshape(imUD1.shape[0],
                           imUD1.shape[1],imUD1.shape[2],imUD1.shape[3])
imgrecon_te_ica = recon_te_ica.reshape(imD1.shape[0],
                         imD1.shape[1],imD1.shape[2],imUD1.shape[3])

# Reconstructed image
fig = plt.figure(figsize = (10, 7.2)) 
fig.add_subplot(121)
plt.title("Original Image",fontsize=15)
plt.imshow(imUD1[idx,:,:])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.add_subplot(122)
plt.title("Reconstructed Image (ICA)",fontsize=15)
plt.imshow(imgrecon_tr_ica[idx,:,:])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("recon_ica"+datasetidx+".png", dpi=600)
plt.show()

# explained variance
print("Sum of explained_variance (PCA)", sum(evr_tr_ica))
fig = plt.figure(figsize = (15, 7.5)) 
ax = plt.axes()
ax.set(facecolor = "white")
plt.title("ICA Explained Variance for Dataset-" + datasetidx,fontsize=25)
plt.ylabel('Explained variance',fontsize=25)
plt.xlabel('Eigen Values',fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.bar(list(range(1,nd+1)),evr_tr_ica)
plt.savefig("evr_ica"+datasetidx+".png", dpi=600)

# reconstruction error
errors_ica = reconstruction_error(imUD1, imgrecon_tr_ica)
print("Shape of reconstruction error on trained training set", 
      np.array(errors_ica).shape)
print("Mean of reconstruction error (ICA)",np.mean(errors_ica))
print("SD of reconstruction error (ICA)",np.std(errors_ica))
print("Median of reconstruction error (ICA)",np.median(errors_ica))
plt.figure(figsize=(20,6))
plt.hist(errors_ica, bins = 50)
plt.xlabel("Reconstruction error",fontsize=22)
plt.ylabel("No of samples",fontsize=22)
plt.show()

# construct the training and testing split for baseline images
(trainX_ica, testX_ica) = train_test_split(Xtr_ica, test_size = 0.15, 
                                           random_state=42)
print("Shape of training set", trainX_ica.shape)
print("Shape of validation set", testX_ica.shape)

# latent space
viz_latspc3D(trainX_ica,testX_ica,Xte_ica,"latent_ica_",datasetidx)

del imgrecon_tr_ica
del imgrecon_te_ica
del recon_tr_ica
del recon_te_ica

############## PCA 
print("===========>>>>> PCA for dimensionality reduction.... ")
pca = PCAmodel()

(Xtr_pca, evr_tr_pca, recon_tr_pca) = pca.pcabuild(imUD1_flat, nd)
(Xte_pca, evr_te_pca, recon_te_pca) = pca.pcabuild(imD1_flat, nd)

imgrecon_tr_pca = recon_tr_pca.reshape(imUD1.shape[0],
                           imUD1.shape[1],imUD1.shape[2],imUD1.shape[3])
imgrecon_te_pca = recon_te_pca.reshape(imD1.shape[0],
                         imD1.shape[1],imD1.shape[2],imUD1.shape[3])

    
fig = plt.figure(figsize = (10, 7.2)) 
fig.add_subplot(121)
plt.title("Original Image",fontsize=15)
plt.imshow(imUD1[idx,:,:])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.add_subplot(122)
plt.title("Reconstructed Image (PCA)",fontsize=15)
plt.imshow(imgrecon_tr_pca[idx,:,:])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("recon_pca"+datasetidx+".png", dpi=600)
plt.show()

# explained variance
print("Sum of explained_variance (PCA)", sum(evr_tr_pca))
fig = plt.figure(figsize = (15, 7.5)) 
ax = plt.axes()
ax.set(facecolor = "white")
plt.title("PCA Explained Variance for Dataset-" + datasetidx,fontsize=25)
plt.ylabel('Explained variance',fontsize=25)
plt.xlabel('Eigen Values',fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.bar(list(range(1,nd+1)),evr_tr_pca)
plt.savefig("evr_pca"+datasetidx+".png", dpi=600)

# reconstruction error
errors_pca = reconstruction_error(imUD1, imgrecon_tr_pca)
print("Shape of reconstruction error on trained training set", 
      np.array(errors_pca).shape)
print("Mean of reconstruction error (PCA)",np.mean(errors_pca))
print("SD of reconstruction error (PCA)",np.std(errors_pca))
print("Median of reconstruction error (PCA)",np.median(errors_pca))
plt.figure(figsize=(20,6))
plt.hist(errors_pca, bins = 50)
plt.xlabel("Reconstruction error",fontsize=22)
plt.ylabel("No of samples",fontsize=22)
plt.show()

# construct the training and testing split for baseline images
(trainX_pca, testX_pca) = train_test_split(Xtr_pca, test_size = 0.15, 
                                           random_state=42)
print("Shape of training set", trainX_pca.shape)
print("Shape of validation set", testX_pca.shape)

# latent space
viz_latspc3D(trainX_pca,testX_pca,Xte_pca,"latent_pca_",datasetidx)

del imgrecon_tr_pca
del imgrecon_te_pca
del recon_tr_pca
del recon_te_pca
del imUD1
del imD1

##############################################################################
############################# ANOMALY DETECTION ##############################
##############################################################################
acc_pca = np.zeros((9,2)) # 9 = rows for nu, 2 = columns for baseline & damage
acc_ica = np.zeros((9,2))
nu = np.round(np.arange(0.1,1,0.1),1)
nn = -1
for ii in nu:
    nn = nn+1
    print("ii=",ii)
    print("The value of nu ==========>>>>>>>>",ii)
    ############################## PCA-SVM ################################
    # implement ocSVM
    pcaocsvm = ocSVM.svmbuild(trainX_pca,ii)
    ## Compute the empirical error 
    ytest_ud_pca = pcaocsvm.predict(testX_pca)
    ytest_d_pca = pcaocsvm.predict(Xte_pca)
    normal_ud_pca = ytest_ud_pca[ytest_ud_pca == 1].size
    anomaly_d_pca = ytest_d_pca[ytest_d_pca == -1].size
    acc_base_pca = normal_ud_pca/testX_pca.shape[0]
    acc_dam_pca = anomaly_d_pca/Xte_pca.shape[0]
    print("(PCA) Number of normal in healthy dataset for nu = " + str(ii) + 
          "====>>>" ,normal_ud_pca)
    print("(PCA) Number of anomaly in damaged dataset for nu = " + str(ii) + 
          "====>>>",anomaly_d_pca)
    print("Testing accuracy of PCA-ocSVM on healthy samples ==>> ",acc_base_pca)
    print("Testing accuracy of PCA-ocSVM on damaged samples ==>> ",acc_dam_pca)
    acc_pca[nn,0] = np.round(acc_base_pca,2)
    acc_pca[nn,1] = np.round(acc_dam_pca,2)
    
    ############################## ICA-SVM #################################
    # implement ocSVM
    icaocsvm = ocSVM.svmbuild(trainX_ica,ii)
    ## Compute the empirical error 
    ytest_ud_ica = icaocsvm.predict(testX_ica)
    ytest_d_ica = icaocsvm.predict(Xte_ica)
    normal_ud_ica = ytest_ud_ica[ytest_ud_ica == 1].size
    anomaly_d_ica = ytest_d_ica[ytest_d_ica == -1].size
    acc_base_ica = normal_ud_ica/testX_ica.shape[0]
    acc_dam_ica = anomaly_d_ica/Xte_ica.shape[0]
    print("(ICA) Number of normal in healthy dataset for nu = " + str(ii) 
          + "====>>>" , normal_ud_ica)
    print("(ICA) Number of anomaly in damaged dataset for nu = " + str(ii) 
          + "====>>>", anomaly_d_ica)
    print("Testing accuracy of ICA-ocSVM on healthy samples ==>> ",acc_base_ica)
    print("Testing accuracy of ICA-ocSVM on damaged samples ==>> ",acc_dam_ica)
    acc_ica[nn,0] = np.round(acc_base_ica,2)
    acc_ica[nn,1] = np.round(acc_dam_ica,2)

# convert arrays into .csv and store for plotting in matlab
np.savetxt("acc_pca"+datasetidx+".csv", acc_pca,fmt='%.2f',delimiter=',')
np.savetxt("acc_ica"+datasetidx+".csv", acc_pca,fmt='%.2f',delimiter=',') 
   
############################# PLOTS ##########################################
fig = plt.figure(figsize = (12, 12))
fig.add_subplot(211)
plt.title("Accuracy of PCA-ocSVM vs nu",fontsize=22)
plt.plot(nu,acc_pca[:,0],'--o',linewidth=2, markersize=12)
plt.plot(nu,acc_pca[:,1],'--^',linewidth=2, markersize=12)
plt.xlabel("nu",fontsize=22)
plt.ylabel("Accuracy",fontsize=22)
plt.xticks(fontsize=20)
plt.yticks([0.2,0.4,0.6,0.8,1.0],fontsize=20)
plt.legend(['Baseline', 'Damage'], loc='best',fontsize=22)
fig.add_subplot(212)
plt.title("Accuracy of ICA-ocSVM vs nu",fontsize=22)
plt.plot(nu,acc_ica[:,0],'--o',linewidth=2, markersize=12)
plt.plot(nu,acc_ica[:,1],'--^',linewidth=2, markersize=12)
plt.legend(['Baseline', 'Damage'], loc='best',fontsize=22)
plt.xticks(fontsize=20)
plt.yticks([0.2,0.4,0.6,0.8,1.0],fontsize=20)
plt.xlabel("nu",fontsize=22)
plt.ylabel("Accuracy",fontsize=22)
fig.tight_layout(pad=1.5)
plt.savefig("acc" + datasetidx + ".png", dpi=600)
 
############################## t-SNE ########################################
from sklearn.manifold import TSNE
perpl = [5,20,30,40,50]
lrate = [10,20,40,80,160,320]
for i in perpl:
    print("Perplexity =", i)
    tsne = TSNE(n_components=3, learning_rate=200, perplexity=i)
    Xtr_tsne = tsne.fit_transform(imUD1_flat)
    Xte_tsne = tsne.fit_transform(imD1_flat)
    
    plt.figure(figsize = (6, 6))
    plt.scatter(Xtr_tsne[:,0], Xtr_tsne[:,1],c='green',marker="^")
    plt.scatter(Xte_tsne[:,0], Xte_tsne[:,1], c='red',marker="o")
    plt.legend(['Baseline-Train', 'Damage-Test'], loc='best',fontsize=12)
    plt.title("t-SNE visualization for Dataset-"+datasetidx+" with Perplexity="
              + str(i) + " (1-2)")
    
    plt.figure(figsize = (6, 6))
    plt.scatter(Xtr_tsne[:,1], Xtr_tsne[:,2],c='green',marker="^")
    plt.scatter(Xte_tsne[:,1], Xte_tsne[:,2], c='red',marker="o")
    plt.legend(['Baseline-Train', 'Damage-Test'], loc='best',fontsize=12)
    plt.title("t-SNE visualization for Dataset-"+datasetidx+" with Perplexity="
              + str(i)+ " (1-3)")
    
    plt.figure(figsize = (6, 6))
    plt.scatter(Xtr_tsne[:,0], Xtr_tsne[:,2],c='green',marker="^")
    plt.scatter(Xte_tsne[:,0], Xte_tsne[:,2], c='red',marker="o")
    plt.legend(['Baseline-Train', 'Damage-Test'], loc='best',fontsize=12)
    plt.title("t-SNE visualization for Dataset-"+datasetidx+" with Perplexity="
              + str(i)+ " (1-3)")
    
    plt.figure(figsize = (6, 6))
    ax = plt.axes(projection='3d')  
    ax.scatter3D(Xtr_tsne[:,0], Xtr_tsne[:,1], Xtr_tsne[:,2],c='green',marker="^")
    ax.scatter3D(Xte_tsne[:,0], Xte_tsne[:,1], Xte_tsne[:,2], c='red',marker="o")
    ax.legend(['Baseline-Train', 'Damage-Test'], loc='best',fontsize=12)
    plt.title("t-SNE visualization for Dataset-"+datasetidx+" with Perplexity="
              + str(i)+" (1-2)")
    ax.view_init(-150, 60) 
    # ax.tick_params(axis='x', labelsize=20)
    # ax.tick_params(axis='y', labelsize=20)
    # ax.tick_params(axis='z', labelsize=20))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()