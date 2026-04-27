"""
A simple implementation of Linear Regression.

Algorithm aim:
1.Input X and Y ✅
2.Normalize the data ✅
3.Train it using SGD ✅
4.Store the weights ✅
5.Accept the new inputs for prediction ✅
6.A proper metric system to evaluate model ❌
===================================================
Goals and implementations for future
[IMP] Use numpy and matrix replacing lists(improved speed) ❌
1.Integrate The Fast Matrix system for distributed evaluation ❌

"""
import random
import math

def epoch(X,Y,W,lr,loss_update):
    '''
    1.Traverse via each X lists
    2.Use the W to get Y predicted
    3.Get error by Y_pred-Y
    4.Update each weight by W(n) = W(n) - error*(learning rate)*x(n)

    '''
    for n, x in enumerate(X):
        y_p = sum(W[i]*float(x[i]) for i in range(len(W)))
        err = y_p - float(Y[n])
        

        for i in range(len(W)):
            W[i] -= loss_update(err,lr,x[i])


class Updater():
    def __init__(self,update_type='RMSE'):
        updateList = {'RMSE':(self.RMSE,self.loss_RMSE),'MAE':(self.MAE,self.loss_MAE)}
        self.update_function = updateList[update_type][0]
        self.loss_func = updateList[update_type][1]

    def get(self):
        return self.update_function,self.loss_func

    def RMSE(self,error,lr,x):
        return lr * error * float(x)
    def MAE(self,error,lr,x):
        if not error == 0:
            sign_error = abs(error)/error
        else:
            return 0 #no change needed
        
        return sign_error*lr*x
    
    def loss_RMSE(self,X,Y,W):
        '''
        Use the RMSE cost function
        C = (Sum(|y_pred-Y_actual|^2)/(total length of dataset))^(1/2)
        '''
        total = 0
        for n, x in enumerate(X):
            y_p = sum(W[i]*float(x[i]) for i in range(len(W)))
            total += (y_p - float(Y[n]))**2
        return (total / len(X))**(1/2)
    
    def loss_MAE(self,X,Y,W):
        '''
        Use the MAE cost function
        C = Sum(|y_pred-Y_actual|)/(total length of dataset)
        '''
        total = 0
        for n, x in enumerate(X):
            y_p = sum(W[i]*float(x[i]) for i in range(len(W)))
            total += abs(y_p - float(Y[n]))
        return total / len(X)

class LinearRegression():
    def __init__(self):
        pass
    def __normalizer(self,X:list,Y:list,mode=0):
        '''
        Methodology:
        1.Calculate the standard deviation of a feature/coloumn
        2.Then normalize each by (data-mean_of_coloumn)/std
        Repeat for all coloumns
        Do this for X and Y
        3.Store the mean and std for each column
        Mode:0==Training mode 1==Testing mode(no not change normalizer params on train data)
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
        y_mean = sum(Y)/len(Y)
        y_std = (sum((y-y_mean)**2 for y in Y)/len(Y))**0.5
        self.Normalizer_column_Y.append((y_mean,y_std))
        Y = [(y - y_mean)/(y_std + 1e-8) for y in Y]

        return X,Y
    
    def __transform(self,X):
        '''
        Transform the X values using the saved transformer data...
        '''
        X_d = self.Normalizer_column_X

        new_X = []

        for n,xi in enumerate(X[:-1]):
            new_X.append((float(xi)-X_d[n][0])/(X_d[n][1]+1e-8))
        
        return new_X
    
    def __detransform(self,Y):
        '''
        Transform the X values using the saved transformer data...
        '''
        Y_mean = self.Normalizer_column_Y[0][0]
        Y_std = self.Normalizer_column_Y[0][1]

        new_Y  = Y*(Y_std+1e-8) + Y_mean
        
        return new_Y


    def fit(self,X:list,Y:list,learning_rate:int=0.0001,max_epoch_lim:int=1000,
              threshold_stop:int=0.001,update_instance=None):
        '''
        1.Noramlize data
        2.Control epoch
        3.Send the data to Epoch function
        '''
        if update_instance is None:
            update_instance = Updater()
            
        #Get loss function and weight updater
        update_function,loss_function = update_instance.get()

        #get loss function from update instance
        X_len = len(X[0])
        Y_len = len(X)
        #initalize the weights no of X value + 1 last one for constant
        W = []

        for wn in range(X_len+1):
            W.append(random.uniform(-0.01, 0.01))

        #1.First normalize X and Y
        X,Y=self.__normalizer(X,Y,mode=0) #run in train mode
        #2.Add an extra colomn for contant full of 1 epoch will auto handle it as a feature
        for i in range(len(X)):
            X[i].append(1)


        #Train with multiple epoch handle the threshold and epoch limit
        loss_last = 0
        for x in range(max_epoch_lim):
            #call the epoch function
            epoch(X,Y,W,learning_rate,update_function)
            #update the loss value
            l = loss_function(X,Y,W)
            delt_l = loss_last-float(l)
            loss_last = l

            #check for thresold
            if threshold_stop > math.sqrt(delt_l**2):
                break

        self.W = W #Keep the weights
        self.loss = loss_last #store the final loss
        
    def predict(self,X:list):
        '''
        pass new data to trained weights.
        normalize x --> use weights to get normalized Y --> denormalize Y
        '''
        X = self.__transform(X) #normalize the values
        X.append(1) #for the constant term
        
        #multiply with W(weights)
        Y_pred = 0
        for n,i in enumerate(X):
            Y_pred += i*self.W[n]

        return self.__detransform(Y_pred)

    def predictList(self,X:list):
        '''
        pass each data to predict() and get data and append it to list and resturn
        '''
        predListOutput = [] #store the output of the predictions
        for dataPoint in X:
            predListOutput.append(self.predict(dataPoint))

        return predListOutput
        




