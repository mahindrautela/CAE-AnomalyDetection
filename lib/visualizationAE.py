from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

class Visualize:
    def viz_predictions(InpImg, decoded, mshow):
        # initialize our list of output images
        outputs = None
        # loop over our number of output samples
        for i in mshow:
            # grab the original image and reconstructed image
            original = (InpImg[i] * 255).astype("uint8")
            recon = (decoded[i] * 255).astype("uint8")
        
            # stack the original and reconstructed image side-by-side
            output = np.hstack([original, recon])

            # if the outputs array is empty, initialize it as the current
            # side-by-side image display
            if outputs is None:
                outputs = output
                # otherwise, vertically stack the outputs
            else:
                outputs = np.vstack([outputs, output])   
        
        return outputs

    def viz_latspc(encoder,trainX,testX_ud,testX_d,axis):
        # axis = [0,1] means plot [z1,z2]
        latent_train = encoder.predict(trainX)
        latent_test_ud = encoder.predict(testX_ud)
        latent_test_d = encoder.predict(testX_d)
        plt.figure(figsize=(6, 6))
        plt.scatter(latent_train[:, axis[0]], latent_train[:, axis[1]],
                    c='green',marker="^")
        plt.scatter(latent_test_ud[:, axis[0]], latent_test_ud[:, axis[1]],
                    c='green',marker="^")
        plt.scatter(latent_test_d[:, axis[0]], latent_test_d[:, axis[1]],
                    c='red',marker="o")
        plt.legend(['Baseline-Train', 'Baseline-Test','Delaminated-Test'], 
                   loc='best',fontsize=18)
        plt.xlabel('z1', fontsize=20)
        plt.ylabel('z2', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()
    
    def viz_latspc3D(encoder,trainX,testX_ud,testX_d,i):
        latent_train = encoder.predict(trainX)
        latent_test_ud = encoder.predict(testX_ud)
        latent_test_d = encoder.predict(testX_d)
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        plt.title('Latent space for Dataset-' + i,fontsize=22)
        ax.scatter3D(latent_train[:,0],latent_train[:,1],latent_train[:,2],
                    c='green',marker="^")
        ax.scatter3D(latent_test_ud[:,0],latent_test_ud[:,1],latent_test_ud[:,2],
                    c='green',marker="o")
        ax.scatter3D(latent_test_d[:,0],latent_test_d[:,1],latent_test_d[:,2],
                    c='red',marker="o")
        plt.legend(['Baseline-Train', 'Baseline-Test','Delaminated-Test'], 
                   loc='best',fontsize=18)
        ax.view_init(-150, 60) 
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='z', labelsize=20)
        savepath = "E:/OneDrive - Indian Institute of Science/PhD-MSR/"\
                   "Unsupervised_SHM/DelamDetect/PythonCodes/results/"
        name = "latentspace_"
        plt.savefig(savepath + name + i + ".png", dpi=600)
        # ax.set_xlabel('Z1', fontsize=15)
        # ax.set_ylabel('Z2', fontsize=15)
        # ax.set_ylabel('Z3', fontsize=15)
        plt.show()

    #---Summarize history for loss        
    def viz_MSEloss(H,axis,i):
        plt.figure(figsize=(20,6))
        plt.plot(H.history['loss'],'-o')
        plt.plot(H.history['val_loss'],'-s')
        plt.title('Reconstruction Loss Curve for Dataset-' + i,fontsize=22)
        plt.ylabel('MSE Loss',fontsize=22)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.xlabel('Number of epochs',fontsize=22)
        plt.legend(['train', 'test'], loc='best',fontsize=22)
        plt.axis(axis)
        savepath = "E:/OneDrive - Indian Institute of Science/PhD-MSR/"\
                   "Unsupervised_SHM/DelamDetect/PythonCodes/results/"
        name = "MSE_"
        plt.savefig(savepath + name + i + ".png", dpi=600)
        plt.show()
        
    #---Summarize history for MAE
    def viz_MAE(H,axis,i):
        plt.figure(figsize=(20,6))
        plt.plot(H.history['mae'],'-o')
        plt.plot(H.history['val_mae'],'-s')
        plt.title('MAE for Dataset-' + i,fontsize=22)
        plt.ylabel('MAE',fontsize=22)
        #plt.grid()
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.xlabel('Number of epochs',fontsize=22)
        plt.legend(['train', 'test'], loc='best',fontsize=22)
        plt.axis(axis)
        plt.show()
        
    #---Summarize history for MAE
    def viz_R2(H,axis,i):
        plt.figure(figsize=(20,6))
        plt.plot(H.history['r_square'],'-o')
        plt.plot(H.history['val_r_square'],'-s')
        plt.title('R^2 for Dataset-' + i,fontsize=22)
        plt.ylabel('R^2',fontsize=22)
        #plt.grid()
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.xlabel('Number of epochs',fontsize=22)
        plt.legend(['train', 'test'], loc='best',fontsize=22)
        plt.axis(axis)
        #plt.savefig("loss_code2.png", dpi=150)
        plt.show()
        
    def viz_thresh_ud(errors_ud,t1,t2,i):
        errors_ud = np.array(errors_ud)
        idx1 = np.arange(0,errors_ud.shape[0],1)       
        plt.figure(figsize=(20,6))
        plt.plot(idx1,errors_ud,'og')
        plt.plot(idx1,t1 * np.ones(errors_ud.shape[0]), '-k')
        plt.plot(idx1,t2 * np.ones(errors_ud.shape[0]), '-b')
        plt.title("Prediction on Dataset-"+ i,fontsize=22)
        plt.xlabel("Sample number", fontsize = 22)
        plt.ylabel("Reconstruction Errors", fontsize = 22)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(["Baseline", 'Threshold-1', 'Threshold-2'], 
                   loc='best',
                   fontsize=22)
        # plt.axis([101,1661,0,0.000005])
        savepath = "E:/OneDrive - Indian Institute of Science/PhD-MSR/"\
                   "Unsupervised_SHM/DelamDetect/PythonCodes/results/"
        name = "threshold_ud_"
        plt.savefig(savepath + name + i + ".png", dpi=600)
        plt.show()
        
    def viz_thresh_d(errors_ud,errors_d,t1,t2,i):
        errors_ud = np.array(errors_ud)
        errors_d = np.array(errors_d)
        errors_all = np.concatenate((errors_ud, errors_d),axis = 0)
        idx2 = np.arange(errors_ud.shape[0],errors_all.shape[0],1)       
        plt.figure(figsize=(20,6))
        plt.plot(idx2,errors_d,'or')
        plt.plot(idx2,t1 * np.ones(errors_d.shape[0]), '-k')
        plt.plot(idx2,t2 * np.ones(errors_d.shape[0]), '-b')
        plt.title("Prediction on Dataset-"+ i,fontsize=22)
        plt.xlabel("Sample number", fontsize = 22)
        plt.ylabel("Reconstruction Errors", fontsize = 22)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(['Delaminated', 'Threshold-1', 'Threshold-2'], 
                   loc='best',
                   fontsize=22)
        #plt.axis([101,1661,0,0.000005])
        savepath = "E:/OneDrive - Indian Institute of Science/PhD-MSR/"\
                   "Unsupervised_SHM/DelamDetect/PythonCodes/results/"
        name = "threshold_d_"
        #plt.savefig(savepath + name + i + ".png", dpi=600)
        plt.show()

    def viz_thresh_all(errors_ud,errors_d,t1,t2,i):
        errors_ud = np.array(errors_ud)
        errors_d = np.array(errors_d)
        errors_all = np.concatenate((errors_ud, errors_d),axis = 0)
        idx1 = np.arange(0,errors_ud.shape[0],1)   
        idx2 = np.arange(errors_ud.shape[0],
                         errors_d.shape[0]+errors_ud.shape[0],1)  
        plt.figure(figsize=(20,6))
        plt.plot(idx1,errors_ud,'og')
        plt.plot(idx2,errors_d,'sr')
        plt.plot(t1 * np.ones(errors_all.shape[0]), '-k')
        plt.plot(t2 * np.ones(errors_all.shape[0]), '-b')
        plt.title("Prediction on Dataset-"+ i,fontsize=22)
        plt.xlabel("Sample number", fontsize = 22)
        plt.ylabel("Reconstruction Errors", fontsize = 22)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(['Baseline', 'Delaminated', 'Threshold-1','Threshold-2'], 
                   loc='best',fontsize=22)
        #plt.axis([0,1661,0,0.005])
        savepath = "E:/OneDrive - Indian Institute of Science/PhD-MSR/"\
                   "Unsupervised_SHM/DelamDetect/PythonCodes/results/"
        name = "threshold_all_"
        plt.savefig(savepath + name + i + ".png", dpi=600)
        plt.show()