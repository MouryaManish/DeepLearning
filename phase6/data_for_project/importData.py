import scipy.io as sio
import numpy as np
import math

#the data has 32 schizophrenic patient and 33 healthy person
#1-16 hp 17-31 sp 32 48 hp 49 65 sp. 16-15-17-17.

def data_invade():
    path = "data_for_project/sortedData.mat"
    #path = "sortedData.mat"
    mat = sio.loadmat(path,squeeze_me=True)
    data = mat.get("Data")
    # retrieve the output patient list 
    output = [] 
    for i in data[:,0]:
        output.append(i[2]-1)
 
    d1 = data[:,1]
    d2 = np.zeros((65,6,91,5))

    for i in range(65):
        for r in range(6):
            if d1[i][r,2][:,:].shape[0]!=91:
                pad = 91 - d1[i][r,2][:,:].shape[0]
                d2[i,r] = np.pad(d1[i][r,2][:,:],((0,pad),(0,0)),mode="constant")
    
    d2 = d2.reshape(65,546,5)
    
    return output,d2


def get_data():

    testing = False
    output,d2 = data_invade()
    # data polishing
    mean = np.mean(d2,axis=(0,1))
    std = np.std(d2,axis=(0,1))
    d2 = (d2 - mean)/std
    #output = np.repeat(output,546)
    hp_data = np.append(d2[0:16],d2[32:49],axis=0)
    sp_data = np.append(d2[16:32],d2[49:65],axis=0)
    hp_output = np.append(output[0:16],output[32:49])
    sp_output = np.append(output[16:32],output[49:65])
    
    #print("\nhp data\n",hp_data.shape)
    #print("\nsp data\n",sp_data.shape)
    #print("\nhp output\n",hp_output.shape)
    #print("\nsp output\n",sp_output.shape)
    
    #     1  2   3   4    5     6     7      8       9   10 
    #   0-3,3-6,6-9,9-12,12-15,15-18,18-21,21-24,24-28,28-31 | 3
 
    # set 2 
    i,j = 0,3
    X_test = hp_data[i:j]
    X_test = np.append(X_test,sp_data[i:j],axis=0)

    y_test = hp_output[i:j]
    y_test = np.append(y_test,sp_output[i:j])


    hp_data = np.delete(hp_data,[i,i+1,i+2],axis=0)
    sp_data = np.delete(sp_data,[i,i+1,i+2],axis=0)
    hp_output = np.delete(hp_output,[i,i+1,i+2])
    sp_output = np.delete(sp_output,[i,i+1,i+2])

        
    X_val = hp_data[0:3]
    X_val = np.append(X_val,sp_data[0:3],axis=0)

    y_val = hp_output[0:3]
    y_val = np.append(y_val,sp_output[0:3])


    hp_data = np.delete(hp_data,[0,1,2],axis=0)
    sp_data = np.delete(sp_data,[0,1,2],axis=0)
    hp_output = np.delete(hp_output,[0,1,2])
    sp_output = np.delete(sp_output,[0,1,2])
    
    
    X_train = hp_data[:]
    X_train = np.append(X_train,sp_data[:],axis=0)

    y_train = hp_output[:]
    y_train = np.append(y_train,sp_output[:])

    permute_1 = np.random.permutation(6)
    permute_11 = np.random.permutation(6)
    permute_2 = np.random.permutation(53)
    X_val = X_val[permute_1]
    X_test = X_test[permute_11]
    X_train = X_train[permute_2]
    y_val = y_val[permute_1]
    y_test = y_test[permute_11]
    y_train = y_train[permute_2]
    
    """ 
    #-------------------------------------
    
    i,j = 31,33
    X_test = hp_data[i:j]
    X_test = np.append(X_test,sp_data[i:j-1],axis=0)

    y_test = hp_output[i:j]
    y_test = np.append(y_test,sp_output[i:j-1])


    hp_data = np.delete(hp_data,[i,i+1,i+2],axis=0)
    sp_data = np.delete(sp_data,[i,j-1],axis=0)
    hp_output = np.delete(hp_output,[i,i+1,i+2])
    sp_output = np.delete(sp_output,[i,j-1])

        
    X_val = hp_data[0:3]
    X_val = np.append(X_val,sp_data[0:3],axis=0)

    y_val = hp_output[0:3]
    y_val = np.append(y_val,sp_output[0:3])


    hp_data = np.delete(hp_data,[0,1,2],axis=0)
    sp_data = np.delete(sp_data,[0,1,3],axis=0)
    hp_output = np.delete(hp_output,[0,1,2])
    sp_output = np.delete(sp_output,[0,1,2])
    
    
    X_train = hp_data[:]
    X_train = np.append(X_train,sp_data[:],axis=0)

    y_train = hp_output[:]
    y_train = np.append(y_train,sp_output[:])
    #print("\ntrain\n ",X_train.shape)
   # print("\ntrain\n ",y_train.shape)
   # print("\ntest\n ",X_test.shape)
   # print("\ntest\n ",y_test.shape)
   # print("\nval\n ",X_val.shape)
   # print("\nval\n ",y_val.shape)
    # changing the shape of the input data
    permute_1 = np.random.permutation(6)
    permute_2 = np.random.permutation(53)
    permute_3 = np.random.permutation(3)



    X_val = X_val[permute_1]
    X_test = X_test[permute_3]
    X_train = X_train[permute_2]
    y_val = y_val[permute_1]
    y_test = y_test[permute_3]
    y_train = y_train[permute_2]
    #--------------------------------------------------
    """ 
    # for testin on large set
    if testing: 
        X_test =X_train.reshape(-1,5)
        y_main_test = y_train[:]
        y_test = np.repeat(y_train,546)

        X_train = X_train.reshape(-1,5)
        X_val = X_val.reshape(-1,5)
        y_main_train = y_train[:]
        y_main_val = y_val[:]
        y_train = np.repeat(y_train,546)
        y_val = np.repeat(y_val,546)

    else:
        X_train = X_train.reshape(-1,5)
        X_test =X_test.reshape(-1,5)
        X_val = X_val.reshape(-1,5)
        y_main_train = y_train[:]
        y_main_test = y_test[:]
        y_main_val = y_val[:]
        y_train = np.repeat(y_train,546)
        y_test = np.repeat(y_test,546)
        y_val = np.repeat(y_val,546)
    

    """
    print("X_val",X_val.shape)  
    print("y_val",y_val.shape)  
    print("y_main_val",y_main_val.shape)  
    print("y_main_val\n",y_main_val)  
    print("X_test",X_test.shape)  
    print("y_test",y_test.shape)  
    print("y_main_test\n",y_main_test.shape)  
    print("y_main_test\n",y_main_test)  
    print("X_train",X_train.shape)  
    print("y_train",y_train.shape)  
    print("y_main_train",y_main_train.shape)  
    print("y_main_train\n",y_main_train)  
    """
    return {
                'X_train':X_train,'y_train':y_train,
                'y_main_train':y_main_train,
                'X_val':X_val, 'y_val':y_val,
                'y_main_val':y_main_val,
                'X_test':X_test,'y_test':y_test,
                'y_main_test':y_main_test
     }
    
get_data()

