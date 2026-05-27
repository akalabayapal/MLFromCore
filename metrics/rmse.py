'''
Calculates RMSE for certain models
'''
import math

def RMSE(modelObject,X,Y):
    loss = 0
    counter = 0
    for prediction in modelObject.predict_stream(X):
        loss += (prediction-Y[counter])**2
        counter += 1

    return math.sqrt(loss/(counter+1))



    
