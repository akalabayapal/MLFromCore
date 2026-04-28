"""
A simple implementation of Binary Logistic Regression using Sigmoid.

Algorithm aim:
1.Input X and Y ✅
2.Normalize the data ✅
3.Train it using Cross Entropy Loss minimisation  ✅
4.Store the weights ✅
5.Accept the new inputs for prediction ✅
6.A proper metric system to evaluate model ❌
===================================================
Goals and implementations for future
[IMP] Use numpy and matrix replacing lists(improved speed) ❌
1.Integrate The Fast Matrix system for distributed evaluation ❌

Implementation logic:
We need to reduce the loss function(cross entropy).Logistic regression loss models the bernoulis trails
P(y 1|0) = y(p) + (1-y)(1-p) p:probabilty of y
Now we need to the decrease the loss L = -log(P(y 1|0))

we take z=m1x1+m2x2+m3x3+....+b 
now to convert it to a proability b/w 1-0 we use sigmoid f(x) y = 1/(1+e^-z)

After simplificaton the updating formula comes out to be: m(n)' = m(n) - (learning_rate)*(y'-y)*x(n)

"""
import random 
import math
from normalizer import Normalizer

def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        return math.exp(z) / (1 + math.exp(z))
def epoch(X,Y,W,lr):
    '''
    1.Traverse via each X lists
    2.Use the W to get Y predicted
    3.Get error by Y_pred-Y
    4.Update each weight by W(n) = W(n) - error*(learning rate)*x(n)

    '''
    for n, x in enumerate(X):
        y_p = sigmoid(sum(W[i]*float(x[i]) for i in range(len(W))))
        eps = 1e-8
        y_p = max(min(y_p, 1 - eps), eps)

        err = y_p - float(Y[n])
        
        for i in range(len(W)):
            W[i] -= lr * err * float(x[i])

def loss(X, Y, W):
    '''
    Use the std cost function
    C = -[ylog(p(y)) + (1-y)log(p(y))]
    '''
    total = 0
    for n, x in enumerate(X):
        y_p = sigmoid(sum(W[i]*float(x[i]) for i in range(len(W))))
        eps = 1e-8
        y_p = max(min(y_p, 1 - eps), eps)
        total += -(Y[n]*math.log(y_p,math.e)+(1-Y[n])*math.log(1-y_p,math.e))
    return total / len(X)

class LogisticRegression():
    def __init__(self):
        self.normalizer = Normalizer()

    def __normalizer(self,X:list,Y:list):
        # '''
        # Methodology:
        # 1.Calculate the standard deviation of a feature/coloumn
        # 2.Then normalize each by (data-mean_of_coloumn)/std
        # Repeat for all coloumns
        # Do this for X and Y
        # 3.Store the mean and std for each column
        # '''
        # self.Normalizer_column_Y = []
        # self.Normalizer_column_X = []

        # #if the model is on predict mode do not chane 
        # #Normalize the X and Y
        
        # for j in range(len(X[0])):
        #     col = [float(X[i][j]) for i in range(len(X))]
        #     mean = sum(col)/len(col)
        #     std = (sum((v-mean)**2 for v in col)/len(col))**0.5
        #     self.Normalizer_column_X.append((mean,std))
        #     for i in range(len(X)):
        #         X[i][j] = (float(X[i][j]) - mean) / (std + 1e-8)
        
        # Y = [float(y) for y in Y]
        

        return self.normalizer.__normalizer(X,Y)
    def __transform(self,X):
        # '''
        # Transform the X values using the saved transformer data...
        # '''
        # X_d = self.Normalizer_column_X

        # new_X = []

        # for n,xi in enumerate(X):
        #     new_X.append((float(xi)-X_d[n][0])/(X_d[n][1]+1e-8))
        
        return self.normalizer.__transform(X)
    
    def __detransform(self,Y:list):
        return self.normalizer.__detransform(Y)
    def fit(self,X:list,Y:list,learning_rate:float=0.0001,max_epoch_lim:int=1000,
              threshold_stop:int=0.001):
        '''
        1.Noramlize data
        2.Control epoch
        3.Send the data to Epoch function
        '''

        X_len = len(X[0])
        Y_len = len(X)
        #initalize the weights no of X value + 1 last one for constant
        W = []

        for wn in range(X_len+1):
            W.append(random.uniform(-0.01, 0.01))

        #1.First normalize X 
        X=self.__normalizer(X,Y)[0] #run in train mode

        #2.Add an extra colomn for contant full of 1 epoch will auto handle it as a feature
        for i in range(len(X)):
            X[i].append(1)

        
        #Train with multiple epoch handle the threshold and epoch limit
        loss_last = 0
        for x in range(max_epoch_lim):
            b2 = epoch(X,Y,W,learning_rate)
            l = loss(X,Y,W)
            delt_l = loss_last-float(l)
            loss_last = l
            if threshold_stop > math.sqrt(delt_l**2):
                break
            #print(loss_last)

        self.W = W #Keep the weights
        self.loss_final = loss_last

    def predict(self,X:list):
        '''
        pass new data to trained weights.
        normalize x --> use weights to get normalized Y --> denormalize Y
        '''

        if not isinstance(X, list) or isinstance(X[0], list):
            raise ValueError("Expected 1D list like [f1, f2,...]. Use predictList() for multiple samples.")
        

        X = self.__transform(X) #normalize the values
        X.append(1) #for the constant term
        
        #multiply with W(weights)
        Y_pred = 0
        for n,i in enumerate(X):
            Y_pred += i*self.W[n]
        
        return sigmoid(Y_pred)
    
    def predictList(self,X:list):
        '''
        1.for each X send to predict().
        2.append to list
        3.return the total list
        '''
        resultList = []
        for x in X:
            resultList.append(self.predict(x))

        return resultList

    

        