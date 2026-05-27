import math

def MAE(modelObject,X,Y):
    loss = 0
    counter = 0
    for prediction in modelObject.predict_stream(X):
        loss += abs(prediction-Y[counter])
        counter += 1

    return loss/(counter+1)
