def R2(modelObj,X,Y):
    if len(Y) == 0 or len(X) != len(Y):
        raise RuntimeError('Model can not be evaluated on no data or the length of X and Y do not match')
    mean = sum(Y)/len(Y)
    counter = 0
    varience = 0
    unexp_var = 0
    for i in modelObj.predict_stream():
        
        unexp_var += (i-Y[i])**2
        varience += (mean-Y[i])**2

    return 1-(unexp_var/varience)
