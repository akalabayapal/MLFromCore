'''
TF-IDF vectorizer

1.Use BOW logic made previously.
2.Get vocab and count the documents containing it to get idf
'''

import math
from collections import Counter

class TFIDFVectorizer():
    def __init__(self):
        self.is_fit = False


    def __buildvocab(self,X:list,sep=' '):
        '''
        loop each of the sentence and split words and return the unique words
        '''
        vocab = set()
        for sentence in X:
            words = sentence.split(sep)
            vocab.update(words)
        return list(vocab)


    def __transformer(self,vocabulary:list,s:str,sep=' '):
        
        if not self.is_fit:
            raise RuntimeError('Data transform requested before fitting the data.')

        transformed_temp = []


        words = s.split(sep)
        word_count = Counter(words)
        for obj in vocabulary:
                transformed_temp.append(word_count.get((obj),0)*self.__idfList[obj])
            
        total = math.sqrt(sum([i**2 for i in transformed_temp]))

        if total == 0:
            return [0.0]*len(vocabulary)

        transformed = []

        for n,i in enumerate(transformed_temp):
            transformed.append((i/total))

        
        return transformed
        
    def fit(self,X:list,sep=' '):
        '''
        1.Get the vocabulary.
        2.Now loop each object count the occurance
        3.store it
        '''
        vocabulary = self.__buildvocab(X,sep)

        self.__vocab = vocabulary #store the vocab so user can use it

        self.__idfList = self.__makeidf(X,sep) #get the idflist
        self.is_fit = True

        self.__sep = sep
        
    def getVocab(self):
        return self.__vocab
    
    def getIDF(self):
        return self.__idfList
    

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
