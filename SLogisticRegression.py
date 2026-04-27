'''
A simple implementation of Multiclass Classifer/Softmax classifier

Implementation logic:
1.Input X and Y
2.Find unique classes in Y
3.Assign a random weight to each class and store it in a list
4.traverse the dataset and for each find out the output get error E

use the weightlist of the correct class and update weights

'''

import math
import random 

def softmax(Zlist):
    '''
    1.Get the scores for all classes.
    2.Return the correct class score
    '''

    s = 0
    v = {}
    for i in Zlist:
        val = math.exp(Zlist[i]) 
        s += val
        v[i] = val

    scores = {}
    for i in v:
        eps = 1e-8 # prevent div by 0 error
        scores[i] = v[i]/(s+eps)

    return scores 



def epoch(X,Y,Wlist,lr):
    '''
    1.Traverse via each X lists
    2.Use the W to get Y predicted
    3.Get error by Y_pred-Y
    4.Update each weight by W(n) = W(n) - error*(learning rate)*x(n)

    '''
    for n, x in enumerate(X):
        zList = {}

        for i in Wlist:

            W = Wlist[i]
            z = sum(W[j]*float(x[j]) for j in range(len(W)))
            zList[i] = z
        
        scores = softmax(zList)

        err = {}

        #make error of correct lass pred-1 else pred-0
        correctClass = Y[n]

        for c in scores:
            #check if correct class
            if c == correctClass:
                err[c]  = scores[c] - 1
            else:
                err[c]  = scores[c] #pred - 0

       #for each of the weight update it

        for e in err:
            #get the correct W
            W = Wlist[e]

            #update the weigts
            for i in range(len(W)):
                 W[i] -= lr * err[e] * float(x[i])
    
    

        # eps = 1e-8
        # y_p = max(min(y_p, 1 - eps), eps)

        # err = y_p - float(Y[n])
        
        # for i in range(len(W)):
        #     W[i] -= lr * err * float(x[i])



class SLogisticRegression():
    def __init__():
        pass
    

    def __getuniqueclass(self,Y) -> list:
        return list(set(Y))
    
    def __makeWeights(self,n):
        W = []
        for wn in range(n+1):
            W.append(random.uniform(-0.01, 0.01))

        return W
    def __normalizer(self,X:list,Y:list):
        '''
        Methodology:
        1.Calculate the standard deviation of a feature/coloumn
        2.Then normalize each by (data-mean_of_coloumn)/std
        Repeat for all coloumns
        Do this for X and Y
        3.Store the mean and std for each column
        '''
        self.Normalizer_column_Y = []
        self.Normalizer_column_X = []

        #if the model is on predict mode do not chane 
        #Normalize the X and Y
        
        for j in range(len(X[0])):
            col = [float(X[i][j]) for i in range(len(X))]
            mean = sum(col)/len(col)
            std = (sum((v-mean)**2 for v in col)/len(col))**0.5
            self.Normalizer_column_X.append((mean,std))
            for i in range(len(X)):
                X[i][j] = (float(X[i][j]) - mean) / (std + 1e-8)
        
        Y = [float(y) for y in Y]
        

        return X,Y
    def __transform(self,X):
        '''
        Transform the X values using the saved transformer data...
        '''
        X_d =self.Normalizer_column_X

        new_X = []

        for n,xi in enumerate(X):
            new_X.append((float(xi)-X_d[n][0])/(X_d[n][1]+1e-8))
        
        return new_X

    def fit(self,X:list,Y:list,learning_rate:int=0.0001,max_epoch_lim:int=1000,
              threshold_stop:int=0.001):
        
        class_Y = self.__getuniqueclass(Y) #get unique classes in the Y

        WList = {} #stores weights for each of the clases

        for i in class_Y:  #for each of the class make Weights
            WList[i] = self.__makeWeights(len(X[0]))
        

        #1.First normalize X 
        X=self.__normalizer(X,Y)[0] #run in train mode

        #2.Add an extra colomn for contant full of 1 epoch will auto handle it as a feature
        for i in range(len(X)):
            X[i].append(1)

        epoch(X,Y,WList,learning_rate)
        

