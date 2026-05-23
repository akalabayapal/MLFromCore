

'''
for each feature get all the data of that feature ans sort it in ascending order
find midpoints
split the data per midpoint store them
for each midpoint find the spit 
calculate gini
find best split
return it
'''

'''
Tree implementation of the decision tree

        TREE
          |(condition)
-----------------------
|                      |
Left                   Right

1.if the deadend is achived store leaf bool for it
2. For traversal...
    -Get data
    -Start of with top node pass the data 1
    - node 1 matches the req field and value and passes it to next node
    - if the node is leaf,final weighted class composition is returned

'''
#Auto unpacking node system..
'''
Once Node is made just call action..node is auto unpacked to verdict...
'''
class Node:
    def __init__(self,isLeaf:bool,value:int=None,field:int=None):
        self.isLeaf = isLeaf
        self.field = field
        self.value = value
        self.left = None
        self.right = None


    
    def setLeftRight(self,left,right):
        self.left = left
        self.right=right
    
    
    def action(self,data):
        if self.isLeaf:
            return self.LeafScore
        
        if self.value > data[self.field]:
            return self.left.action(data)
        else:
            return self.right.action(data)
    
    def SetLeafScore(self,scorearr:list):
        if self.isLeaf == False:
            raise RuntimeError('Can not assign LeafScore to non leaf node...')
        
        self.LeafScore:list = scorearr #set the probability of each class in a list/array







class DecisionTree:
    def __init__(self):
        pass

    def shortviafeature(self,data,featureno,y):

        temp_data = {}
        featureListValue = []
        shorted = []
        shortedDict = {}
        shortedExpanded = []
        midPoints = []
        Y_linker = []

        Index_linker = []  # <--- 1. ADD THIS TO TRACK ORIGINAL ROW INDICES

        for n,d in enumerate(data):
            featureValue = d[featureno]

            if featureValue in temp_data:
                temp_data[featureValue].append(n)

            else:
                temp_data[featureValue] = [n]
                featureListValue.append(featureValue)

        isFirst = True
        featureListValue.sort()
        size = 0

        for v in featureListValue:
            for j in temp_data[v]:
                value = data[j][featureno]
                Y_linker.append(y[j])
                shortedExpanded.append(value)
                Index_linker.append(j)  # <--- 2. STORE THE EXACT ROW INDEX HERE
                try:
                    shortedDict[value]
                except:
                    if isFirst:
                        shorted.append(value)
                        shortedDict[value] = {}
                        size += 1
                        isFirst = False
                    else:
                        shorted.append(value)
                        shortedDict[value] = {}
                        midPoints.append((shorted[size-1]+value)/2)
                        size += 1

        del shorted
        del shortedDict
        del featureListValue

        return midPoints,shortedExpanded,Y_linker,temp_data,Index_linker # <--- 3. RETURN IT AS WELL
    
    def ginicalc(self,data_y,left_size,right_size,y_u):

        total_size = left_size+right_size
        category_index = {}

        for n,category in enumerate(y_u):
            category_index[category] = n

        Y_u_len = len(y_u)

        counterLeft=[0]*Y_u_len
        counterRight = [0]*Y_u_len

        for n,i in enumerate(data_y):
            if n<=(left_size-1):
                counterLeft[category_index[i]] += 1
            else:
                counterRight[category_index[i]] += 1

        PLeft = 0
        for i in counterLeft:
            PLeft += ((i/left_size))**2
        giniLeft = (left_size/total_size)*(1-PLeft)


        Pright = 0
        for i in counterRight:
            Pright += ((i/right_size))**2

        giniRight = (right_size/total_size)*(1-Pright)


        return giniLeft+giniRight
    

    def splitfinder(self,featureno,storeddata,Y_u):
        best_gini = 2
        split_best = None

        # <--- 1. EXTRACT index_linker FROM storeddata
        points,x_values,y_values,linker,index_linker = storeddata[featureno] 
        left = []
        right:list = y_values
        right_index = 0

        left_x = []
        right_x:list = x_values

        size_right = len(y_values)
        size_left = 0

        for point in points:
            for i in range(right_index,size_right):
                if i > size_right - 1:
                    break
                if right_x[i] < point:
                    left.append(right[right_index])
                    left_x.append(right_x[right_index])
                    right_index+=1
                    size_left += 1
                else: 
                    break

            gini = self.ginicalc(y_values,size_left,size_right-size_left,Y_u)

            if gini < best_gini or gini == 0:
                best_gini = gini
                # <--- 2. ADD index_linker TO THE split_best TUPLE
                split_best = (point,x_values,size_left,y_values,best_gini,index_linker) 

        return split_best

                
            
    def uniqueY(self,y):
        return set(y)

    def RecursiveBuilder(self,x,y,prevNode:Node=None,depth=0,max_depth=100,Y_u:list=[]):
        depth  = depth+1
        dataStorage = []
        if depth == max_depth:
            node = Node(True)
            node.SetLeafScore(y)
            return node 

        for f in range(self.UYlen):
            dataStorage.append(self.shortviafeature(x,f,y))

        bestsplitginni = 2
        bestSplit= ()
        bestfeature = 0

        for j in range(self.UYlen):
            splitdata = self.splitfinder(j,dataStorage,Y_u)
            # <--- 1. UPDATE INDEX TO -2 BECAUSE bestSplit NOW HAS 6 ELEMENTS
            if splitdata and splitdata[-2] < bestsplitginni: 
                bestsplitginni = splitdata[-2]
                bestSplit=splitdata
                bestfeature = j

        if bestSplit == ():
            #end tree here
            node = Node(True)
            node.SetLeafScore(y)
            return node 

        pivot = bestSplit[2]
        index_linker = bestSplit[5]  # <--- 2. GET THE INDEX LINKER

        # <--- 3. SLICE INDICES DIRECTLY AND GENERATE THE NEW SUBSETS CLEANLY
        indices_left = index_linker[:pivot]
        indices_right = index_linker[pivot:]

        x_left = [x[j] for j in indices_left]
        y_left = [y[j] for j in indices_left]
        x_right = [x[j] for j in indices_right]
        y_right = [y[j] for j in indices_right]

        # ... Rest of your node creation logic remains the same ...
        # 1. Instantiate the local non-leaf node for the current split
        node = Node(False, bestSplit[0], bestfeature)

        # 2. Build sub-trees normally without tracking parents 
        # (Each recursive call returns its own newly generated branch/leaf)
        left_child = self.RecursiveBuilder(x_left, y_left, depth=depth, max_depth=max_depth,Y_u=Y_u)
        right_child = self.RecursiveBuilder(x_right, y_right, depth=depth, max_depth=max_depth,Y_u=Y_u)

        # 3. Attach the children directly to this node
        node.setLeftRight(left_child, right_child)

        # 4. Return THIS node so its parent node can link to it seamlessly
        return node


    def fit(self,X,Y,max_depth=100):
        self.uY = self.uniqueY(Y)
        self.UYlen = len(self.uY)
        self.builder = self.RecursiveBuilder(X,Y,max_depth=max_depth,Y_u=self.uY)

    def __predict(self,kernel,X):
        output = self.builder.action(X)

        return kernel(self.uY,output)
    
    def predict(self,kernel,X:list):
        output = []
        for data in X:
            output.append(self.__predict(kernel=kernel,X=data))

        return output
    
    def predict_stream(self,kernel,X:list):
        for data in X:
            yield self.__predict(kernel,data)



class Kernels():

    @staticmethod
    def Voting(Yu:list,output:list):

        counter = [0]*len(Yu)
        for n,k in enumerate(Yu):
            counter[n] = output.count(k)
        
        return counter
    
    @staticmethod
    def normalizedvoting(Yu:list,output:list):
        length = len(output)
        counter = [0]*len(Yu)
        for n,k in enumerate(Yu):
            counter[n] = output.count(k)/length
        
        return counter
    
    @staticmethod
    def regression(Yu:list,output:list):
        length = len(output)
        counter = [0]*len(Yu)
        for n,k in enumerate(Yu):
            counter[n] = (output.count(k)/length)*k
        
        return sum(counter)





