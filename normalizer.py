class Normalizer():
    def __init__(self):
        pass

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
        X_d = self.Normalizer_column_X

        new_X = []

        for n,xi in enumerate(X):
            new_X.append((float(xi)-X_d[n][0])/(X_d[n][1]+1e-8))
        
        return new_X