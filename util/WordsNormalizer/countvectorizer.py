'''
Implementation of bag of words..

1. Take list of sentences.
2.Split by separator.
3.build vocabulary
4.for each list count the number of the word and store it
'''
from collections import Counter

class CountVectorizer():
    def __init__(self):
        self.is_fit = False


    def __buildvocab(self,X:list,sep=' '):
        '''
        loop each of the sentence and split words and return the unique words
        '''

        vocab = []
        vc = {}

        for i in X:
            objects = i.lower().split(sep)

            for obj in objects:
                try:
                    k = vc[obj]
                except KeyError:
                    vocab.append(obj)
                    vc[obj] = {}
                    
        
        return vocab


    def __transformer(self,s:str,sep=' '):

        transformed = [[]]
        #transformed.extend([0]*self.__vocablen)


        words = s.lower().split(" ")

        words_count = Counter(words)
        
        for obj in words_count.keys():
                #print(obj)
                pos = self.__dictvocab[obj]
                transformed.append(words_count.get(obj,0))
                transformed[0].append(pos)

            
        return transformed
        
    def fit(self,X:list,sep=' '):
        '''
        1.Get the vocabulary.
        2.Now loop each object count the occurance
        3.store it
        '''
        vocabulary = self.__buildvocab(X,sep)

        self.__vocab = vocabulary #store the vocab so user can use it
        self.__vocablen = len(vocabulary)
        self.__dictvocab = {} #store the word and its position in list so that we directly put the value the position
        for n,word in enumerate(vocabulary):
            self.__dictvocab[word] = n


        self.is_fit = True
        self.__sep = sep
        
    def getVocab(self):
        return self.__vocab
    
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

