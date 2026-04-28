'''
TF-IDF vectorizer

1.Use BOW logic made previously.
2.Get vocab and count the documents containing it to get idf
'''

import math
from collections import Counter
from util.vocabgen import *

class TFIDFVectorizer():
    def __init__(self):
        self.is_fit = False



    def __transformer(self,vocabulary:list,s:str,sep=' '):
        
        if not self.is_fit:
            raise RuntimeError('Data transform requested before fitting the data.')

        transformed_temp = []
#transformed_temp.append(word_count.get((obj),0)*self.__idfList[obj])

        transformed_temp = [[]]
        total = 0 #total value

        words = s.lower().split(sep)

        words_count = Counter(words)
        
        for obj in words_count.keys():
                v = words_count.get(obj,0)*self.__idfList[obj]
                transformed_temp[0].append(self.__dictvocab[obj])
                transformed_temp.append(v)

                total += v**2
            
        total = total**(1/2)

        if total == 0:
            return [0.0]*len(vocabulary)

        transformed = [transformed_temp[0]]

        for n,i in enumerate(transformed_temp[1:]):
            transformed.append((i/total))

        
        return transformed
        
    def fit(self,X:list,sep=' '):
        '''
        1.Get the vocabulary.
        2.Now loop each object count the occurance
        3.store it
        '''
        vocabulary = self.vocabObj.fit(X,sep)
        
        self.__vocab = self.vocabObj.__vocab #store the vocab so user can use it
        self.__vocablen = self.vocabObj.__vocablen
        self.__dictvocab = self.vocabObj.__dictvocab

        self.__idfList = self.__makeidf(X,sep=sep)
        
        self.is_fit = True
        self.__sep = sep
    
    def getIDF(self):
        return self.__idfList
    
    def getVocab(self):
        return self.__vocab
    
    def isfit(self):
        return self.is_fit

    def __makeidf(self,X,sep):

        p = {}

        for i in X:
            words_in_doc = set(i.split(sep)) # Use a set to count a word only once per doc
            for w in words_in_doc:
                if w in p:
                    p[w] += 1
                else:
                    p[w] = 1


        # 2. Convert counts to IDF scores using the log formula
        n_samples = len(X)
        idf_dict = {}
        for word, count in p.items():
            # Using the standard smoothed formula: log(N / df) + 1
            idf_dict[word] = math.log(n_samples / count) + 1

        return idf_dict


    def transform(self,X:list):
        '''
        input must be list of strings
        '''
        if not self.is_fit:
            raise RuntimeError('Data transform requested before fitting the data.')
        X_transformed = []

        for i in X:
            if i.strip() == '':
                raise ValueError('The transformation content must not be empty.')
            X_transformed.append(self.__transformer(self.__vocab,i,self.__sep))

        return X_transformed

    def fit_transform(self,X:list,sep=' '):
        self.fit(X=X,sep=sep)
        return self.transform(X)
    
    def deCompress(self,X_compressed:list):
        '''
        Decompress the compressed TF-IDF vector to normal long form
        '''
        #make sure the data is fit already...
        if not self.is_fit:
            raise RuntimeError('Decompression failed as no data has been fit.')
        #check the structure first...
        if len(X_compressed) == 0:
            raise IndexError('The compressed stucture must have atleast length of 1.')
        if not (type(X_compressed)==list and type(X_compressed[0])==list):
            raise TypeError('The given compressed structure does not match with the vectorizer supported stucture.')
        
        decompressed = [0.0]*self.__vocablen #make the mainframe of the decompressed strucure
        for n,i in enumerate(X_compressed[0]):
            decompressed[i] = X_compressed[1+n]

        return decompressed
            


