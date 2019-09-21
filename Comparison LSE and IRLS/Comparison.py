
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.close('all') #close any open plots

""" ======================  Function definitions ========================== """

def plotData(x1,t1,x2=None,t2=None,x3=None,t3=None,legend=[]):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'bo') #plot training data
    if x2 is not None:
        p2 = plt.plot(x2, t2, 'g') #plot true value
    if x3 is not None :
        p3 = plt.plot(x3, t3, 'r') #plot training data

    #add title, legend and axes labels
    plt.ylabel('t') #label x and y axes
    plt.xlabel('x')

    plt.ylim(-2,2)
    plt.xlim(-4.5,4.5)
    
    if x2 is None :
        plt.legend((p1[0]),legend)
    if x3 is None :
        plt.legend((p1[0],p2[0]),legend)
    else:
        plt.legend((p1[0],p2[0],p3[0]),legend)

def plotData_2(x1,t1,x2=None,t2=None,legend=[]):
    '''plotData(x1,t1,x2,t2,legend=[]): Generate a plot of the 
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'g') #plot training data
    if x2 is not None :
        p2 = plt.plot(x2, t2, 'r') #plot true value

    #add title, legend and axes labels
    plt.xlabel('M Values (For K=0.03)') #label x and y axes
    plt.ylabel('Sum Of Absoulte Error ')
    
    if x2 is None :
        plt.legend((p1[0]),legend)
    else:
        plt.legend((p1[0],p2[0]),legend)

def plotData_3(x1,t1,legend=[]):
    plt.plot(x1,t1,'ro', label=legend)
    plt.xlabel('K Values (For M=10)') 
    plt.ylabel('Sum Of Absoulte Error ')
    legend = plt.legend(loc='upper center', shadow=True, fontsize='x-small')

      
def fitdataLS(x,t,M):
    '''fitdataLS(x,t,M): Fit a polynomial of order M to the data (x,t) using LS''' 
    #This needs to be filled in
    X=np.array([x**m for m in range(M+1)]).T
    w=np.linalg.inv(X.T@X)@X.T@t
    return w
        
def fitdataIRLS(x,t,M,k):
    '''fitdataIRLS(x,t,M,k): Fit a polynomial of order M to the data (x,t) using IRLS''' 
    #This needs to be filled in
    X = np.array([x**m for m in range(M+1)]).T

    #Intializing w paramters using LS
    wPrev = np.linalg.inv(X.T@X)@X.T@t
    w = getWdataIR(X,t,k,wPrev)
    return w

def getWdataIR(X,t,k,wPrev):
    #Intializing B (diagonal matrix)
    diag_vec = abs(t.T - wPrev.T@X.T)
    for i in range(diag_vec.size):
        if diag_vec[i] <= k:
            diag_vec[i] = 1
        else :
            diag_vec[i] = k/diag_vec[i]
    
    B = np.diag(diag_vec)
    w = np.linalg.inv(X.T@B@X)@X.T@B@t

    #Iteration applied to converge B with threshold (w-wPrev < 0.001)
    if sum(abs(w-wPrev)) >= 0.0001:
        w1=getWdataIR(X,t,k,w)
        
    else:
        return w
    return w

""" ======================  Variable Declaration ========================== """
M =  10 #regression model order
k = 0.03 #Huber M-estimator tuning parameter

""" =======================  Load Training Data ======================= """
data_uniform = np.load('TrainData.npy')
x1 = data_uniform[:,0]
t1 = data_uniform[:,1]

    
""" ========================  Train the Model ============================= """
wLS = fitdataLS(x1,t1,M) 
wIRLS = fitdataIRLS(x1,t1,M,k) 


""" ======================== Load Test Data  and Test the Model =========================== """

"""This is where you should load the testing data set. You shoud NOT re-train the model   """
test_data = np.load('TestData.npy')
x2 = test_data[:,0]
t2 = test_data[:,1]

X1 = np.array([x2**m for m in range(wLS.size)]).T
esty_LS = X1@wLS

X2 = np.array([x2**m for m in range(wIRLS.size)]).T
esty_IRLS = X2@wIRLS

""" ========================  Plot Results ============================== """

""" This is where you should create the plots requested """

#Plot 1
plotData(x1, t1, x2, esty_LS, x2, esty_IRLS, ['Training Data', 'Estimated\nLeast Square', 'Estimated\nH-Estimation'])

#Plot 2
abs_error_LS = []
abs_error_IRLS = []
M_values = np.arange(1,18,1)
for _m in M_values:
    _wLS = fitdataLS(x1,t1,_m)
    _X1 = np.array([x2**m for m in range(_wLS.size)]).T
    _esty_LS = _X1@_wLS
    abs_error_LS.append(sum(abs(_esty_LS-t2)))

    _wIRLS = fitdataIRLS(x1,t1,_m,k)
    _X2 = np.array([x2**m for m in range(_wIRLS.size)]).T
    _esty_IRLS = _X2@_wIRLS
    abs_error_IRLS.append(sum(abs(_esty_IRLS-t2)))

plotData_2(M_values, abs_error_LS, M_values, abs_error_IRLS, ['Estimated\nLeast Square for K=0.03', 'Estimated\nH-Estimation for K=0.03'] )


#Plot 3
K_values = np.arange(0.01,1,0.05)
huber_error = []

for _k in K_values:
    _wIRLS = fitdataIRLS(x1,t1,M,_k)
    _X2 = np.array([x2**m for m in range(_wIRLS.size)]).T
    _esty_IRLS = _X2@_wIRLS
    huber_error.append(sum(abs(_esty_IRLS-t2)))

plotData_3(K_values, huber_error, 'Estimated\nH-Estimation for M = 10')
