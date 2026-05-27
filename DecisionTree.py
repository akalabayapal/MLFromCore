import csv
import random

class Node:
    def __init__(self, isLeaf: bool, value: int = None, field: int = None):
        self.isLeaf = isLeaf
        self.field = field
        self.value = value
        self.left = None
        self.right = None

    def setLeftRight(self, left, right):
        self.left = left
        self.right = right
    
    def action(self, data):
        if self.isLeaf:
            return self.LeafScore
        
        if self.value > data[self.field]:
            return self.left.action(data)
        else:
            return self.right.action(data)
    
    def SetLeafScore(self, scorearr: list):
        if self.isLeaf == False:
            raise RuntimeError('Can not assign LeafScore to non leaf node...')
        self.LeafScore: list = scorearr


class DecisionTree:
    def __init__(self):
        self.STOP_SIG = 'STOP'
        self.RUN_SIG = 'RUN'

    def shortviafeature(self, data, index_linker, type_scan, pivot, featureno, y, init_u=None, end_u=None):
        # Establish exact scanning limits for this specific branch window
        if type_scan == 'left' or type_scan == 'left_unchecked':
            init = init_u
            end = pivot
        elif type_scan == 'right':
            init = pivot
            end = end_u

        temp_data = {}
        featureListValue = []
        shorted = []
        shortedDict = {}
        shortedExpanded = []
        midPoints = []
        Y_linker = []
        Index_linker = []  

        for n in range(init, end):
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
                Index_linker.append(j)  
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
        
        return midPoints, shortedExpanded, Y_linker, temp_data, Index_linker
    

    def splitfinder(self, featureno, storeddata, Y_u):
        best_gini = 2
        split_best = None

        points, x_values, y_values, linker, index_linker = storeddata[featureno] 
        left = []
        right: list = y_values
        right_index = 0
        total_size = len(y_values)

        left_x = []
        right_x: list = x_values

        size_right = len(y_values)
        size_left = 0

        signal = self.RUN_SIG
        if len(points) == 0:
            signal = self.STOP_SIG

        len_Yu = len(Y_u)
        counterLeft = [0]*len_Yu
        counterRight = [0]*len_Yu
        category_index = {category: n for n, category in enumerate(Y_u)}

        for y in y_values:
            idx = category_index[y]
            counterRight[idx] += 1

        for point in points:
            for i in range(right_index, size_right):
                if i > size_right - 1:
                    break
                if right_x[i] < point:
                    left.append(right[right_index])
                    left_x.append(right_x[right_index])

                    categoryIndex = category_index[y_values[right_index]]
                    counterRight[categoryIndex] -= 1
                    counterLeft[categoryIndex] += 1

                    right_index += 1
                    size_left += 1
                    size_right -= 1
                else: 
                    break

            if size_left == 0 or size_right == 0:
                continue

            PLeft = sum((count / size_left) ** 2 for count in counterLeft)
            giniLeft = (size_left / total_size) * (1 - PLeft)
    
            PRight = sum((count / size_right) ** 2 for count in counterRight)
            giniRight = (size_right / total_size) * (1 - PRight)
    
            gini = giniLeft + giniRight

            if size_right <= self.minsamples:
                signal = self.STOP_SIG

            if gini < best_gini or gini == 0:
                best_gini = gini
                # Keep tracking size_left to locate the relative boundary shift
                split_best = (point, x_values, size_left, y_values, best_gini, index_linker) 

        return split_best, signal

    def uniqueY(self, y):
        return set(y)

    def fit(self, X, Y, max_depth=100, min_samples=10):
        self.features_unique = len(X[0])
        self.minsamples = min_samples
        self.uY = self.uniqueY(Y)
        self.UYlen = len(self.uY)
        
        # Instantiate a proper mutable tracking index array
        global_indices = list(range(len(X)))
        
        self.builder = self.RecursiveBuilder(
            x=X,
            y=Y,
            index_linker=global_indices,
            pivot=len(X),               
            type_scan='left_unchecked',  
            init_u=0,            
            end_u=len(X),               
            max_depth=max_depth,
            Y_u=self.uY
        )

    def RecursiveBuilder(self, x, y, index_linker, pivot, type_scan, init_u, end_u, depth=0, max_depth=100, Y_u:list=[]):
        # Base Case: Stop recursion if boundaries collapse
        if init_u is None or end_u is None or init_u >= end_u:
            node = Node(True)
            fallback_idx = init_u if (init_u is not None and init_u < len(index_linker)) else 0
            node.SetLeafScore([y[index_linker[fallback_idx]]])
            return node

        depth = depth + 1
        dataStorage = {}
        
        if depth == max_depth:
            node = Node(True)
            branch_y = [y[idx] for idx in index_linker[init_u:end_u]]
            node.SetLeafScore(branch_y)
            return node

        # Gather sorted features inside local node limits
        for f in range(self.features_unique):
            dataStorage[f] = self.shortviafeature(x, index_linker, type_scan, pivot, f, y, init_u, end_u)

        if type_scan == 'left_unchecked':
            type_scan = 'left'

        bestsplitginni = 2
        bestSplit = ()
        bestfeature = 0

        for j in range(self.features_unique):
            splitdata, signal = self.splitfinder(j, dataStorage, Y_u)
            if splitdata and splitdata[-2] < bestsplitginni: 
                bestsplitginni = splitdata[-2]
                bestSplit = splitdata
                bestfeature = j

        if bestSplit == ():
            node = Node(True)
            branch_y = [y[idx] for idx in index_linker[init_u:end_u]]
            node.SetLeafScore(branch_y)
            return node 

        # --- THE MATHEMATICAL CORRECTION ---
        # The true index pivot location inside the global array must be 
        # relative to where our starting window boundary (init_u) is positioned!
        size_left = bestSplit[2]
        new_pivot = init_u + size_left       
        
        # Write back the sorted mutation indices into our shared array tracker
        mutated_branch_indices = bestSplit[5]
        index_linker[init_u:end_u] = mutated_branch_indices

        if signal == self.STOP_SIG:
            node = Node(True)
            branch_y = [y[idx] for idx in index_linker[init_u:end_u]]
            node.SetLeafScore(branch_y)
            return node 

        current_start = init_u
        current_end = end_u

        node = Node(False, bestSplit[0], bestfeature)

        left_child = self.RecursiveBuilder(
            x, y, 
            index_linker=index_linker, 
            pivot=new_pivot, 
            type_scan='left', 
            init_u=current_start, 
            end_u=new_pivot,  
            depth=depth, 
            max_depth=max_depth, 
            Y_u=Y_u
        )
        
        right_child = self.RecursiveBuilder(
            x, y, 
            index_linker=index_linker, 
            pivot=new_pivot, 
            type_scan='right', 
            init_u=new_pivot, 
            end_u=current_end, 
            depth=depth, 
            max_depth=max_depth, 
            Y_u=Y_u
        )

        node.setLeftRight(left_child, right_child)
        return node

    def __predict(self, kernel, X):
        output = self.builder.action(X)
        return kernel(self.uY, output)
    
    def predict(self, kernel, X: list):
        output = []
        for data in X:
            output.append(self.__predict(kernel=kernel, X=data))
        return output

class Kernels:
    @staticmethod
    def Voting(Yu: list, output: list):
        counter = [0]*len(Yu)
        for n, k in enumerate(Yu):
            counter[n] = output.count(k)
        return dict(zip(Yu, counter))
    
    @staticmethod
    def StrictClass(Yu: list, output: list):
        list_Yu = list(Yu)
        counter = [0]*len(Yu)
        for n, k in enumerate(Yu):
            counter[n] = output.count(k)

        maxiclass = None
        maxi = -1
        for j in range(len(Yu)):
            if maxi < counter[j]:
                maxi = counter[j]
                maxiclass = list_Yu[j]
        return maxiclass


