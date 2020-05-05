#!/usr/bin/env python
# coding: utf-8

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.datasets import load_iris


def fit_NeuralNetwork(X_train,y_train,alpha,hidden_layer_sizes,epochs):
    #Enter implementation here
    #init weights
    inti_weights = []
    #return values
    weights = []
    err = []
    
    #average error per ne epoches
    average_sample = 0
     
    #add one column representing the bias terms to the data set 
    my_ones = np.ones((np.size(X_train, 0),1))
    X_train = np.hstack((my_ones,X_train))
    
    #get the y values for specific data set
    y_value = y_train.flatten()     
    y_value = y_value[0]
    
     
    #forward propagation result
    X = []
    S = []
    #backpropagation result
    g = []
    #errorpersample result
    #eN = 0
    #update weight result
    nW = []
    # index
    i = 0

      
    #get total number of col which is the total number of data points
    num_col = np.size(X_train, 1)
    #get total number of rows which is the total number of data points
    num_row = np.size(X_train, 0)
    
    
    #init weights of layer number one by using input x
    my_matrix = np.array([[0,0],[0,0]])
    my_matrix = np.full((num_col, hidden_layer_sizes[0]), 0.1)
    inti_weights.append(my_matrix)
    i = 1
    num_col = hidden_layer_sizes[0]
    
    #init all weights for all nodes in all layers
    while i < len(hidden_layer_sizes):
               
        my_matrix = np.array([[0,0],[0,0]])
        my_matrix = np.full((num_col + 1, hidden_layer_sizes[i]), 0.1)
        inti_weights.append(my_matrix)
        num_col = hidden_layer_sizes[i]
        i = i + 1
        if i == len(hidden_layer_sizes):
            my_matrix = np.array([[0,0],[0,0]])
            my_matrix = np.full((num_col + 1, 1), 0.1)
            inti_weights.append(my_matrix)
            
    
    #loop epochs number of time over the datasets
    for e in range (epochs):
      i = 1
      while i  <= num_row:
          
        #get an "i" data point   
        data_point = X_train[i-1:i,:]
        data_point = np.transpose(data_point)
        
        #get the y value for the data point
        y_value = y_train[i-1:i]
        
        #apply forward, backward and update for the specific datapoint
        X,S = forwardPropagation(data_point, inti_weights) 
        g  =  backPropagation(X, y_value, S, inti_weights) 
        nW = updateWeights(inti_weights,g,alpha) 
        inti_weights = nW
        average_sample = average_sample + errorPerSample(X,y_value)
        i = i + 1
        
        
      #append the average error in error list 
      err.append((average_sample/num_row))
      average_sample = 0
      #X_train = np.random.shuffle(X_train)
     
    #assign the final updated weights to the weights list    
    weights = inti_weights
    return err, weights
    
def errorPerSample(X,y_n):
    #Enter implementation here
    eN = 0  
    #get the error for the last x_output
    eN = errorf(X[len(X) - 1],y_n)     
    return eN
 
def forwardPropagation(x, weights):
    #Enter implementation here
    
    #get lenght of weights
    list_length = len(weights)
    i = 0
    #return values X represent the output and s the summation
    X = []
    S = []
    
    #current output
    my_out = x

    X.append(x)
    i = 0
    
    #loop over the list of wright for different layers
    while i < list_length:
        
       #if the weight for the last layer , then use outpud fuction not activation 
       if i == len(weights) - 1:
            x_vector = np.array([])
            s = np.dot(np.transpose(weights[i]),my_out )
            s = np.asmatrix(s)
            if np.size(s, 1) > 1:
              s = np.transpose(s)
            S.append(s)
            s = np.squeeze(np.asarray(s))
            
           
            x_vector= np.hstack((x_vector,outputf(s)))
            X.append(x_vector)
            my_out = np.transpose(x_vector)
        
       else: 
        x_vector = np.array([1]) 
       # taje the dot product then convert to a matrix then make sure it is a vertical matrix just to make it easier for calculation
        s = np.dot(np.transpose(weights[i]),my_out)
        s = np.asmatrix(s)
        if np.size(s, 1) > 1:
            s = np.transpose(s)
        
        S.append(s)     
        s = np.squeeze(np.asarray(s))
        x_vector = x_vector.flatten()
       
        
        
        
        #if the weight is not for the last layer , then use activation fuction  
        for elem in s:
          
           x_vector = np.hstack((x_vector,activation(elem)))    
        X.append(x_vector)
        my_out = np.transpose(x_vector)
                              
       i += 1                       
    # return output and summation for different layers for different nodes
    return X,S     
        

def backPropagation(X,y_n,s,weights):
    #Enter implementation here
    g = []
    #list for backword message
    g_back= []
    
    
    #used as temp array to add reuslt in it before adding it to the list backword message
    result_vector = np.array([])
    
    
    last_s = np.squeeze(np.asarray(s[len(s) - 1]))
    #calculate the backword message for the last layer aka the output node
    result_vector= np.hstack((result_vector,derivativeError(X[len(X) - 1],y_n) * derivativeOutput(last_s)))
            
    result_vector = np.asmatrix(result_vector)
    
    #insert first because the calculation begins at the end
    g_back.insert( 0,result_vector)
    
    #get the prevoius backpropagation message to use it to calculate ither backword messages recurssivly
    previous_backPropagation = derivativeError(X[len(X) - 1],y_n) * derivativeOutput(s[len(s) - 1])
    
    i = len(weights) - 1
    #take the second last summation matrix instead of the last because the last was used to calculate the last backword propagation layer
    ss = len(s) - 2
    #loop throug all summation vectors for different layers to caluclate the backword messages
    while ss >= 0:
        result_vector = np.array([])   
        s_vector = s[ss]
        w_matrix = weights[i]
        result = 0
                
        num_row = np.size(s_vector, 0)     
        #index for weight for specific node
        k = 2
        #index for s vector values
        ee = 1
        #loop through different element of s vector
        for e in range (num_row):
            
            #take the first row of weight vector to calculate the backword message
             w_vector = w_matrix[k-1:k,:]
            
             num_col = np.size(w_vector, 1)            
             vv = 1
             v = 1
             
             #looop throug weight elements of weight vector to calculate the summation to get the backword message
             for h in range (num_col):
                 
                 value1 = w_vector[:,v-1:v]                
                 value1 = np.squeeze(np.asarray(value1))    
                 value2 = s_vector[ee-1:ee]
                 value2 = np.squeeze(np.asarray(value2))
                             
                 value3 = previous_backPropagation[vv-1:vv]
                 value3 = np.squeeze(np.asarray(value3))

                 
                 #funtion to calculate the summation to get the previous backword message
                 result= result + (value3 * value1* derivativeActivation(value2))
                 v = v + 1
                 vv = vv + 1
                 my_array = np.array([result])
             
               #temp array to add the backword messages for different nodes for specific layer
             result_vector = np.hstack((result_vector,my_array))  
                        
             k = k + 1
             ee = ee + 1
             result = 0 
            
         # make the temp array the previous backword message to calculate the next backword message
        previous_backPropagation = result_vector
               
        # insert the backword message to g)back list to use it to calculate the error gradient 
        result_vector = np.asmatrix(result_vector)
        result_vector = np.transpose(result_vector)
     
        g_back.insert( 0,result_vector) 
        result_vector = np.array([])
        ss = ss - 1
        i = i - 1
    # loop though backword message list to calculate the error gradients
    for l in range (len(g_back)):
        
        X_matrix = np.asmatrix(X[l])
        if l > 0:
            X_matrix = np.transpose(X_matrix)        
        #function to calculate the error gradient
        my_result =  np.dot(X_matrix,np.transpose(g_back[l]))       
        #add the calculated gradient to error gradeint list
        g.append(my_result)     
        my_result = 0       
    return g
def updateWeights(weights,g,alpha):
    #Enter implementation here
    nW = []
    
    i = 0
    
    #loop over the list of weights which includes all weights of different layers
    while i < len(weights):
        #applying update function of the weights
        nW.append(weights[i] - alpha * (g[i]))
        i = i + 1
      
    return nW
def activation(s):
    #Enter implementation here
    #function of activation fuction which is relu function
  return np.maximum(0,s)

def derivativeActivation(s):
    #Enter implementation here
    
    
    #derivative of relu function depend on value of summation ,s.
    if (s > 0):
       return 1
    if (s <= 0):
       return 0

def outputf(s):
    #Enter implementation here
    
    #output function which is the logistic regression
    return 1 / (1 + np.exp(-s))

def derivativeOutput(s):
    #Enter implementation here
    
    #derivative of output function which is logistic regression
    return( np.exp(-s)/  ( 1 + np.exp(-s)  *  1 + np.exp(-s) ))


def errorf(x_L,y):
    #Enter implementation here
    
    #init the result of error function
    result = 0
    
    #depend of vlaue of y , different error function will be used  ( log loss function)
    if y == 1:
        result = - np.log(x_L)
    if y == -1:
        result = - np.log(1 - x_L)
    return result    

def derivativeError(x_L,y):
    #Enter implementation here
    
    #depend on value of y choose the different  derivative log loss function 
    if y == 1:
        return - ((1 / x_L) * 1)
    if y == -1:
        return - ((1 / 1-x_L) * -1)
    

def pred(x_n,weights):
    #Enter implementation here
    
    #apply forward prop to get the values of summations of different layers and outputs values for different layers
    x,s = forwardPropagation(x_n, weights)
    
    #covert the vector or matrix into a real number value to compare it with the threshold
    out_x = np.squeeze(x[len(x)- 1])
    
    #if the value of x_output larger or equal to the o.5, threshold, then it belongs to class 1
    if  out_x >= 0.5:
        return 1
    #if the value of x_output less than the threshold then it belongs to class -1 
    if out_x  < 0.5:
        return -1
        

    
    
def confMatrix(X_train,y_train,w):
    #Enter implementation here
    
    
    
    
    #initialize the confusion matrix to zeros
    my_confmatrix = np.array([[0,0],[0,0]])
    
    #index to access different data point
    i = 1
    
    #get total number of rows which is the total number of data points
    num_rows = np.size(X_train, 0)
    
    #add new column to x_training matrix with one values
    my_ones = np.ones((np.size(X_train, 0),1))
    X_train = np.hstack((my_ones,X_train))
    
    #loop to iterate through different data point
    while i  <= num_rows:
      
     #get an "i" data point   
     data_point = X_train[i-1:i,:]
     
     #convert data point to a vector
     data_point = np.transpose(data_point)
     
     #get the y value for the data point
     y_value = y_train[i-1:i]
     y_value = y_value[0]
     
     #convert y value to a vector
     y_value = y_value.flatten()
     
     #get the y value as a number
     y_value = y_value[0]
     
     #get the predicted value of y based on "w"
     predicted_value = pred(data_point,w)
     
     #count of the total number of points from the training set that have been correctly classified to be class −1 by the linear classifier defined by w
     if predicted_value == -1 and y_value == -1:
         
       my_confmatrix[0][0] = my_confmatrix[0][0] + 1
      
     #count of the total number of points from the training set that are in class 1 but are classified by w to be in class −1   
     elif predicted_value == 1 and y_value == -1: 
         
       my_confmatrix[0][1] = my_confmatrix[0][1] + 1
       
     #count of the total number of points from the training set that have been correctly classified to be class 1 by the linear classifier defined by w  
     elif predicted_value == -1 and y_value == 1:
         
       my_confmatrix[1][0] = my_confmatrix[1][0] + 1
       
     #count of the total number of points from the training set that are in class −1 but are classified by w to be in class 1  
     elif predicted_value == 1 and y_value == 1: 
         
       my_confmatrix[1][1] = my_confmatrix[1][1] + 1    
       
     i = i + 1
    
    #return the confusion matrix
    return my_confmatrix      
     
    

def plotErr(e,epochs):
    #Enter implementation here
    
    #make an array from zero to epochs
    epoches_list = np.arange(epochs)
    
    #convert the array to list
    epoches_list.tolist()
    
    #label x axis to epochs
    plt.xlabel("epochs")
    
    #label y axis to error
    plt.ylabel("error")
    
    #plot epoches list and error list to the graph
    plt.plot(epoches_list,e)
       
    #show the graph      
    plt.show()   
    
def test_SciKit(X_train, X_test, Y_train, Y_test):
    #Enter implementation here
    
    #create MLPClassifier object
    mlp = MLPClassifier()
    
    #fit the training point to the model
    mlp.fit(X_train,Y_train)
    
    #get prediction based on our model
    predictions = mlp.predict(X_test)
    
    
    #return confusion matrix for this model
    return confusion_matrix(Y_test,predictions)
     
    

def test():
    
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)
    
    for i in range(80):
        if y_train[i]==1:
            y_train[i]=-1
        else:
            y_train[i]=1
    for j in range(20):
        if y_test[j]==1:
            y_test[j]=-1
        else:
            y_test[j]=1
        
    err,w=fit_NeuralNetwork(X_train,y_train,1e-2,[30, 10],100)
    
    plotErr(err,100)
    
    cM=confMatrix(X_test,y_test,w)
    
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)

test()
