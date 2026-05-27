import csv
import random

'''
Simple Implementation of Decision Tree


for each feature get all the data of that feature ans sort it in ascending order
1find midpoints
2.split the data per midpoint store them
3.for each midpoint find the spit 
4.calculate gini
5.find best split
6.return it

Tree implementation of the decision tree

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
import random

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



class _DecisionTree:
    def __init__(self):
        self.STOP_SIG = 'STOP'
        self.RUN_SIG = 'RUN'

    def shortviafeature(self,data,index_linker,type_scan,pivot,featureno,y,init_u=None,end_u=None):

        if type_scan == 'left':
            init = 0
            end = pivot
        elif type_scan == 'right':
            init = pivot
            end = len(index_linker)
        elif type_scan == 'left_unchecked':
            init = init_u
            end = end_u

        temp_data = {}
        featureListValue = []
        shorted = []
        shortedDict = {}
        shortedExpanded = []
        midPoints = []
        Y_linker = []

        Index_linker = []  # <--- 1. ADD THIS TO TRACK ORIGINAL ROW INDICES

        for n in range(init,end):
            d = data[index_linker[n]]
            featureValue = d[featureno]

            if featureValue in temp_data:
                temp_data[featureValue].append(index_linker[n])

            else:
                temp_data[featureValue] = [index_linker[n]]
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
    

    def splitfinder(self,featureno,storeddata,Y_u):
        best_gini = 2
        split_best = None

        # <--- 1. EXTRACT index_linker FROM storeddata
        points,x_values,y_values,linker,index_linker = storeddata[featureno] 
        left = []
        right:list = y_values
        right_index = 0
        total_size = len(y_values)

        left_x = []
        right_x:list = x_values

        size_right = len(y_values)
        size_left = 0

        if len(points) == 0:
            signal = self.STOP_SIG

        '''Efficiency fix:This we store the counter here update it with the sliding window in runtime by shifting the counter to the left as needed...

        Calculate gini directly here using the counter..

        Need: decreases the algo time complexity from O(N^2) --> O(N)
        '''
        len_Yu = len(Y_u)
        counterLeft = [0]*len_Yu
        counterRight = [0]*len_Yu
        category_index = {category: n for n, category in enumerate(Y_u)}

        for y in y_values:
            idx = category_index[y]
            counterRight[idx] += 1


        for point in points:
            for i in range(right_index,size_right):
                if i > size_right - 1:
                    break
                if right_x[i] < point:
                    left.append(right[right_index])
                    left_x.append(right_x[right_index])

                    #add 1 to left counter for the particular class and substract 1 from the right....
                    categoryIndex = category_index[y_values[right_index]]
                    counterRight[categoryIndex] -= 1
                    counterLeft[categoryIndex] += 1


                    right_index+=1
                    size_left += 1
                    size_right -= 1

                else: 
                    break

            # 4. Inline Gini calculation using our running trackers (NO LOOPS OVER DATA!)
            PLeft = sum((count / size_left) ** 2 for count in counterLeft)
            giniLeft = (size_left / total_size) * (1 - PLeft)
    
            PRight = sum((count / size_right) ** 2 for count in counterRight)
            giniRight = (size_right / total_size) * (1 - PRight)
    
            gini = giniLeft + giniRight
            
            signal = self.RUN_SIG

            if size_right <= self.minsamples:
                signal = self.STOP_SIG
            


            if gini < best_gini or gini == 0:
                best_gini = gini
                # <--- 2. ADD index_linker TO THE split_best TUPLE
                split_best = (point,x_values,size_left,y_values,best_gini,index_linker) 

        return split_best,signal

                
            
    def uniqueY(self,y):
        return set(y)

    def fit_bootstrap(self, X, Y, global_bootstrap_list, start_pos, end_pos, max_depth=100, min_samples=10, k_random_features=10):
        """
        Clean entry point from the forest. Passes the global list and boundaries 
        directly into the recursive builder without any list slicing.
        """
        self.features_unique = len(X[0])
        self.minsamples = min_samples
        self.uY = self.uniqueY(Y)
        self.UYlen = len(self.uY)
        
        # Inject the global pointer array with explicit start and end boundary indexes
        self.builder = self.RecursiveBuilder(
            x=X,
            y=Y,
            index_linker=global_bootstrap_list,
            pivot=end_pos,               # Root pivot starts at the end of the initial window
            type_scan='left_unchecked',  # Directs shortviafeature to use init_u and end_u
            init_u=start_pos,
            end_u=end_pos,
            random_features_max=k_random_features,
            depth=0,
            max_depth=max_depth,
            Y_u=self.uY
        )

    def RecursiveBuilder(self, x, y, index_linker, pivot, type_scan, init_u, end_u, random_features_max=10, depth=0, max_depth=100, Y_u:list=[]):
        """
        Perfected recursive tree builder using localized window boundaries 
        to isolate left and right child mutations in the global index tracker array.
        """
        
        # 1. BASE CASE: Boundary Checkpoint
        # If the window has collapsed or has no data points left, stop and return a leaf
        if init_u is None or end_u is None or init_u >= end_u:
            node = Node(True)
            # Safe fallback score assignment
            fallback_idx = init_u if (init_u is not None and init_u < len(index_linker)) else 0
            node.SetLeafScore([y[index_linker[fallback_idx]]])
            return node

        depth = depth + 1
        dataStorage = {}
        
        # 2. BASE CASE: Maximum Depth Reached
        if depth == max_depth:
            node = Node(True)
            # Gather training target labels strictly within our current window bounds
            branch_y = [y[idx] for idx in index_linker[init_u:end_u]]
            node.SetLeafScore(branch_y)
            return node

        # 3. FEATURE SUBSAMPLING
        random_features = []
        for j in range(random_features_max):
            random_features.append(random.randint(0, self.features_unique - 1))

        # 4. SCAN DATA AND COMPUTE MIDPOINTS
        # Route logic using your existing 'left_unchecked' scanning strategy
        if type_scan == 'left_unchecked':
            for f in random_features:
                dataStorage[f] = self.shortviafeature(x, index_linker, type_scan, pivot, f, y, init_u, end_u)
            # Switch scan type to standard 'left' for subsequent recursive descendant layers
            type_scan = 'left'
        else:
            for f in random_features:
                dataStorage[f] = self.shortviafeature(x, index_linker, type_scan, pivot, f, y)

        # 5. FIND THE OPTIMAL SPLIT
        bestsplitginni = 2
        bestSplit = ()
        bestfeature = 0

        for j in random_features:
            splitdata, signal = self.splitfinder(j, dataStorage, Y_u)
            if splitdata and splitdata[-2] < bestsplitginni: 
                bestsplitginni = splitdata[-2]
                bestSplit = splitdata
                bestfeature = j

        # 6. BASE CASE: No Valid Split Found
        if bestSplit == ():
            node = Node(True)
            branch_y = [y[idx] for idx in index_linker[init_u:end_u]]
            node.SetLeafScore(branch_y)
            return node 

        # 7. CRITICAL SPLIT CONFIGURATION
        new_pivot = bestSplit[2]       # This is where splitfinder tells us to split the current block
        index_linker = bestSplit[5]    # Regain reference to the globally managed index list

        # 8. BASE CASE: Stopping Signal Triggered (e.g., min_samples condition)
        if signal == self.STOP_SIG:
            node = Node(True)
            branch_y = [y[idx] for idx in index_linker[init_u:end_u]]
            node.SetLeafScore(branch_y)
            return node 

        # 9. CAPTURE LAYER BOUNDARIES
        # Explicitly freeze the start and end of our current active segment 
        # before letting children shuffle the contents of the shared index_linker list.
        current_start = init_u
        current_end = end_u

        # 10. NODE CREATION AND RECURSION
        # Initialize the non-leaf internal routing decision node
        node = Node(False, bestSplit[0], bestfeature)

        # Build the Left Subtree: operates from current_start up to the new_pivot position
        left_child = self.RecursiveBuilder(
            x, y, 
            index_linker=index_linker, 
            pivot=new_pivot, 
            type_scan='left', 
            init_u=current_start, 
            end_u=new_pivot, 
            random_features_max=random_features_max, 
            depth=depth, 
            max_depth=max_depth, 
            Y_u=Y_u
        )
        
        # Build the Right Subtree: operates from the new_pivot position up to current_end
        right_child = self.RecursiveBuilder(
            x, y, 
            index_linker=index_linker, 
            pivot=new_pivot, 
            type_scan='right', 
            init_u=new_pivot, 
            end_u=current_end, 
            random_features_max=random_features_max, 
            depth=depth, 
            max_depth=max_depth, 
            Y_u=Y_u
        )

        # Link children to the current decision layer
        node.setLeftRight(left_child, right_child)

        return node


    def fit(self, X, Y, global_bootstrap_list, start_pos, end_pos, max_depth=100, min_samples=10, k_random_features=10):
        self.features_unique = len(X[0])
        self.minsamples = min_samples
        self.uY = self.uniqueY(Y)
        self.UYlen = len(self.uY)
        
        # We pass the entire global list, but we tell the recursive builder 
        # to ONLY look between start_pos and end_pos!
        self.builder = self.RecursiveBuilder(
            x=X,
            y=Y,
            index_linker=global_bootstrap_list,
            pivot=end_pos,               # The pivot boundary starts at the end of the window
            type_scan='left_unchecked',  # 'left_unchecked' naturally handles the start_pos/end_pos math!
            init_u=start_pos,            # Setting the exact starting point in the global list
            end_u=end_pos,               # Setting the exact ending point in the global list
            random_features_max=k_random_features,
            max_depth=max_depth,
            Y_u=self.uY
        )

    def predict(self,kernel,X):
        output = self.builder.action(X)

        return kernel(self.uY,output)
    

class Kernels:

    @staticmethod
    def Voting(Yu:list,output:list):

        counter = [0]*len(Yu)
        for n,k in enumerate(Yu):
            counter[n] = output.count(k)
        
        return counter
    
    @staticmethod
    def StrictClass(Yu:list,output:list):
        list_Yu = list(Yu)
        counter = [0]*len(Yu)
        for n,k in enumerate(Yu):
            counter[n] = output.count(k)

        maxiclass = None
        maxi = 0
        for j in range(len(Yu)):
            if maxi <= counter[j]:
                maxi = counter[j]
                maxiclass = list_Yu[j]
        
        return maxiclass
    
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
        if length == 0:
            return -1
        counter = [0]*len(Yu)
        for n,k in enumerate(Yu):
            counter[n] = (output.count(k)/length)*k
        
        return sum(counter)

class _RandomForestBase:
    def __init__(self):
        pass

    def sampling(self,num_trees,X):

        len_X = len(X)
        random_datapoints =  random.choices(range(len_X), k=num_trees * len_X)

        return random_datapoints

    def fit(self,numtrees, x, y, k_random_features,min_tree_samples,tree_max_depth=100):
        self.Y_unique = list(set(y))

        self.numberTrees = numtrees
        self.forest:list[_DecisionTree] = []
        # One single flat list of integers created once in memory
        samples_selected = self.sampling(numtrees, x)
        len_dataset = len(x)

        for i in range(numtrees):
            o = _DecisionTree()
            # We calculate the boundaries for this specific tree
            init = len_dataset * i
            final = init + len_dataset 


            # WE PASS THE SAME GLOBAL LIST, BUT WITH BOUNDARY MARKERS!
            o.fit(
                X=x, 
                Y=y, 
                global_bootstrap_list=samples_selected, 
                min_samples=min_tree_samples,
                max_depth=tree_max_depth,
                start_pos=init, 
                end_pos=final, 
                k_random_features=k_random_features
            )
            self.forest.append(o)
    
class RandomForestClassifier(_RandomForestBase):

    def __predict(self,X):
        counter = [0]*len(self.Y_unique)
        counterPointer = 0
        category_index = {}

        for tree in self.forest:
            pred = tree.predict(Kernels.StrictClass,X)
            try:
                key = category_index[pred]
                counter[key] += 1
            except:
                category_index[pred] = counterPointer
                # print(counterPointer)
                # print(counter)
                counter[counterPointer] = 1
                counterPointer += 1

        predicted = {}
        for i,category in zip(counter,category_index.keys()):
            predicted[category] = i/self.numberTrees

        return predicted

    def predict(self,X):
        results = []
        for x in X:
            results.append(self.__predict(x))
        return results
    
    def predict_stream(self,X):
        for x in X:
            yield self.__predict(x)

class RandomForestRegression(_RandomForestBase):
    def __predict(self,X):
        counter = 0

        for tree in self.forest:
            counter += tree.predict(Kernels.regression,X)
        
        return counter/self.numberTrees


    def predict(self,X):
        results = []
        for x in X:
            results.append(self.__predict(x))
        return results
    
    def predict_stream(self,X):
        for x in X:
            yield self.__predict(x)
        






        



