import scipy.io as sio
import numpy as np

def data_invade():
    #path = "data_for_project/sortedData.mat"
    path = "sortedData.mat"
    mat = sio.loadmat(path,squeeze_me=True)
    data = mat.get("Data")
    # retrieve the output patient list 
    output = [] 
    for i in data[:,0]:
        output.append(i[2]-1)

    output = np.repeat(output,6)

    d1 = data[:,1]
    d2 = np.zeros((d1.shape[0]*6,1,5001,2))
    count = 0 
    for p in range(65):
        	for r1 in range(6):
	        	for r2 in range(5001):
			        for c2 in range(2):
                                        d2[count,0,r2,c2] = d1[p][r1,1][r2,c2]
		        count +=1


    return output,d2

def data_polish(d2):
    # considring the highest width(1024) and height(756) of all the images
    d3 = np.zeros((d2.shape[0],1,756,1024),dtype=np.uint32)
    for im in range(390):
        for c in range(1):
            for r1 in range(5001):
                x = round(d2[im,c,r1,0])
                y = round(d2[im,c,r1,1])   
                if((x < 1024 and x >= 0) and (y < 756 and y >= 0)):
                    x=int(x)
                    y=int(y)
                    d3[im,c,y,x] += 1  
    
    return d3

def getRegionalCount(d3,width,height):
    assert 1024%width == 0, "divide error by getRegionalCount for wcount"
    assert 756%height == 0, "divide error by getRegionalCount for hcount"
    wcount = 1024//width
    hcount = 756//height
    d4 = np.zeros((d3.shape[0],1,hcount,wcount),dtype = np.uint32)
    wl =[]
    hl=[]
    for im in range(d3.shape[0]):
        for c in range(1):
            for h1 in range(hcount):
                for w1 in range(wcount):
                    d4[im,c,h1,w1] = np.sum(d3[im,c,h1 * height:h1*height+height,w1 * width:w1*width+width])
    return d4

def get_data():
    
    output,d2 = data_invade()
    d3 = data_polish(d2)
    width = 16 #32,16
    height =14  #28,14
    d4 = getRegionalCount(d3,width,height)
    # padding data for use during max pooling and conv. to maintain shape
    d4 = np.pad(d4,((0,0),(0,0),(0,1),(0,0)),mode="constant")
    # seperating the image no. 15 and 20 for testing purpose(total 130,these images are never been exposed to the modal).
    #images were already sorted via  matlab 
    l1 = [True,True,False,True,True,False]
    l1 = np.tile(l1,65)
    l1 = np.logical_not(l1)
    X_test = d4[l1]
    y_test = output[l1]
    l1 = np.logical_not(l1) 
    trainX = d4[l1]
    trainY = output[l1]
    # creating 20 validation set and 240 for training set
    X_val= trainX[240:260]
    y_val = trainY[240:260]
    X_train = trainX[:240]
    y_train = trainY[:240]
    """
    l1 = np.random.choice(260,size=20,replace=False)
    l1.sort()
    X_val = np.zeros((20,1,d4.shape[2],d4.shape[3]))
    y_val = np.zeros(20)
    
    for i in range(20):
            X_val[i] = trainX[l1[i]]
            y_val[i]= trainY[l1[i]]


    X_train = np.delete(trainX,l1,0)
    y_train = np.delete(trainY,l1,0)     
    """        
    return {
            'X_train':X_train,'y_train':y_train,
            'X_val':X_val, 'y_val':y_val,
            'X_test':X_test,'y_test':y_test
     }

