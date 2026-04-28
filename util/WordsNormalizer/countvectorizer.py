'''
Implementation of bag of words..

1. Take list of sentences.
2.Split by separator.
3.build vocabulary
4.for each list count the number of the word and store it
'''
from collections import Counter
from util.vocabgen import *

class CountVectorizer():
    def __init__(self):
        self.vocabObj = Vocabgen()
        self.is_fit = False


    def __transformer(self,s:str,sep=' ',isBinary=False):

        transformed = [[]]
        #transformed.extend([0]*self.__vocablen)


        words = s.lower().split(" ")

        if not isBinary:
            words_count = Counter(words)
        
        for obj in words_count.keys():
                #print(obj)
                pos = self.__dictvocab[obj]
                if isBinary:
                    transformed.append(1)
                else:
                    transformed.append(words_count.get(obj,0))
                transformed[0].append(pos)

            
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

        self.is_fit = True
        self.__sep = sep
        
    def getVocab(self):
        return self.__vocab
    
    def isfit(self):
        return self.is_fit
    
    def transform(self,X:list):
        '''
        input must be list of strings
        '''
        if not self.is_fit:
            raise RuntimeError('Data transform requested before fitting the data.')
        
        X_transformed = []

        for i in X:
            X_transformed.append(self.__transformer(i,self.__sep))

        return X_transformed

    def fit_transform(self,X:list,sep=' '):
        self.fit(X=X,sep=sep)
        return self.transform(X)

    def deCompress(self,X_compressed:list):
        '''
        Decompress the compressed BOW vector to normal long form
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