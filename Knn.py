'''
An simple implementation of KNN(K-nearest neighbours)

#it uses lazy learning stores all data and then finds the nearest k and votes them or avarages them
for classification and regression problems respectively.

Algorithm aim:
1.Input X and Y 
2.Normalize the data 
3.Store the whole data
4.Use distances
    - Eucledian distance
    - Cosine similarity
    - Manhattan distance

5.Implement approximative methods
6.Test out LARM centeroid representation method for knn approximation
'''
import math

def dot_product(x,y):
    s=0
        
        
    for pos,r in enumerate(y):
        diff = r*x[pos]
        s+=diff

    return s

class Distance():
    '''
    Extracted the distance as a separtate class this will help user to test out there distance function
    using helper functions
    '''
    @staticmethod
    def eucledian(X,target):
        '''
        We avoid taking square root as it is wastage of processing power as a>b <=> root(a)>root(b)
        in domain as y=root(x) is an increasing function.
        '''

        s = 0

        for pos,r in enumerate(target):
            diff = (r-X[pos])**2
            s+=diff

        return s

    @staticmethod
    def manhattan(X,target):
        '''
        we take mod of distance of each axis and take sum
        '''
        s=0

        for pos,r in enumerate(target):
            diff = abs(r-X[pos])
            s+=diff

        return s
    
    @staticmethod
    def cosine_sim(X,target):
        '''
        We take the cosine angle b/w two vectors also 1-cosine_sim to maintain the knn logic..so that
        more similar gets less value and the less similar gets more value
        '''

        s = dot_product(X,target)

        mag_X = sum([i**2 for i in X])**(1/2)
        mag_target = sum([i**2 for i in target])**(1/2)

        inverse_sim = 1 - (s/(mag_X*mag_target))

        return inverse_sim
    

class Kernel():

    '''
    KnearestPoints->tuple (distance,y)
    '''
    @staticmethod
    def uniform():
        '''
        Equal priority to all the K points
        '''
        def exec(KnearestPoints:list):
            return [1]*len(KnearestPoints) #equal weights to all points
        
        return exec
    
    @staticmethod
    def inversedistance(eps=1e-8):
        '''
        w = 1/(d+epsillon) #divide by zero prevention by eps(must be very small generally)
        '''

        def exec(KnearestPoints:list):
            W=[]

            #loop through the K points and get the weights

            for d in KnearestPoints:
                distance = d[0] 

                W.append(1/(eps+distance))

            return W
        
        return exec
    
    @staticmethod
    def gauss(decayCostant):
        '''
        formula:w=e^(-x/d^2)
        '''
        def exec(KnearestPoints:list):
            W = []

            for d in KnearestPoints:
                distance = d[0]

                w = math.exp(-distance/(2*decayCostant**2))
                W.append(w)

            return W
        
        return exec
    
    @staticmethod
    def exp():
        def exec(KnearestPoints):
            W = []

            for d in KnearestPoints:
                distance = d[0]
    
                w = math.exp(-distance)
                W.append(w)
    
            return W
        
        return exec
        

    @staticmethod
    def polyinversedstance(power:int,eps=1e-8):
        def exec(KnearestPoints:list):
            W=[]

            #loop through the K points and get the weights

            for d in KnearestPoints:
                distance = d[0] 

                W.append(1/(eps+distance)**power)

            return W
        
        return exec
    
    @staticmethod
    def hardcuttoff(cutoff:int):
        
        
        #loop through the K points and get the weights
        def exec(KnearestPoints:list):
            W=[]

            for d in KnearestPoints:
                distance = d[0] 

                if distance >=cutoff:
                    W.append(0)

                else:
                    W.append(1)

            return W
        
        return exec
        

        

class Knn():
    def __init__(self):
        self.task_enum = ['classification','regression']

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
    
    def __detransform(self,Y):
        '''
        Transform the X values using the saved transformer data...
        '''
        Y_mean = self.Normalizer_column_Y[0][0]
        Y_std = self.Normalizer_column_Y[0][1]

        new_Y  = Y*(Y_std+1e-8) + Y_mean
        
        return new_Y
    

    def getNearestK(self,X,distanceMetric=None):
        X=self.__transform(X) #normalize data

        if distanceMetric == None:
            distanceMetric = Distance.eucledian

        #for all stored data find distance
        distance = []
        for n,data in enumerate(self.X):
            d = distanceMetric(data,X)
            distance.append((d, self.Y[n]))

        distance.sort(key=lambda x: x[0])      
        
        return distance[0:self.K]

    def __predictR(self,X,distanceMetric=None,kernel=Kernel.uniform()):
        '''
        Use Knn as a regression model

        1.Iterate via data store the distance:pos
        2.Sort the list(dict keys) in ascending
        3.Get the 1st K points from list
        4.Take avarage of Y values
        '''
        
        distance = self.getNearestK(X,distanceMetric)

        kernel_weights = kernel(distance)
        y = [y[1] for y in distance]

        d_sum = dot_product(y,kernel_weights)

        w_sum = sum(kernel_weights)

        avg_d = (d_sum/w_sum)

        return self.__detransform(avg_d)
    
    def __predictC(self,X,distanceMetric=None,kernel=Kernel.uniform()):
        '''
        Use Knn as a classification model

        1.Iterate via data store the distance:pos
        2.Sort the list(dict keys) in ascending
        3.Get the 1st K points from list
        
        '''
        distance = self.getNearestK(X, distanceMetric)
        weights = kernel(distance)

        store = {}

        for i, d in enumerate(distance):
            y = d[1]
            w = weights[i]

            if y in store:
                store[y] += w
            else:
                store[y] = w

                e_sum = sum(store.values())

        total = sum(store.values())

        probs = {}
        for k in store:
            probs[k] = store[k] / (total + 1e-8)

        return probs

    def predict(self,X:list,distanceMetric=None,kernel=Kernel.uniform()):
        if self.traintype == 'regression':
            self.__predictR(X,distanceMetric,kernel)
        elif self.traintype == 'classification':
            self.__predictC(X,distanceMetric,kernel)

    def fit(self,X:list,Y:list,K:int,traintype='regression'):
        '''
        Normalize and store the data
        '''
        if not traintype in self.task_enum:
            raise TypeError('The given train type is not supported.') 
        
        self.K = K
        self.traintype = traintype
        
        if traintype == 'regression':
            self.X,self.Y = self.__normalizer(X,Y)
        
        else:
            self.X = self.__normalizer(X,Y)[0] #do not transform Y in the classifcation




