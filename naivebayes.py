'''
Simple implementation of Naive Bayes(MULTINOMIAL)


Implementation logic:
    1.Take the vectors and vocabulary as input
    2.count the number of occurance of object/word N for a class M
    3.get probability for object N for class M(use laplace smoothening) apply log instantly and store them
    

Algorithm methodology:
    1.use the probabilities stored apply P(class) = sum(P(N)) N is word present in input
    2.Take max or apply the kernels to get scroes
'''
from util.vocabgen import Vocabgen
import math
class freqObj():
    def __init__(self,vocablen):
        self.total = 0
        self.Y={}
        self.distinct = 0
        self.vocablen = vocablen
        self.prior = 0
    
    def addprior(self):
        self.prior += 1

    def addy(self,obj,freq):
        if obj in self.Y:
            self.Y[obj] += freq
        else:
            self.Y[obj] = freq
            self.distinct += 1

        self.total += freq

    def getProbability(self,obj):
        
        #use laplace smoothening
        count = self.Y.get(obj, 0)

        return (count+1)/(self.total+self.vocablen)
            


class NaiveBayes():
    
    def __init__(self):
        pass

    
    def fit(self,vocab:Vocabgen,X:list,Y:list):
        '''
        #DISCLAMER:fit only supports compressed BOW,or related structures(see BOW implementation) for performance optimzation

        1.find out the unique classes
        2.Make list of  to store the freqency of obj and the obj in dict freqPbj
        3.also simulatanously store the sum for total words in each type in freqObj
        4.no need to calculate the probability for each now let predict take freq and do division(to reduce computation)
        '''
        freq_list = {}

        vocab_len = vocab.getVocabParams()[0]


        for data,label in zip(X,Y):
            #check if label exists if not make it
            if not label in freq_list:
                freq_list[label] = freqObj(vocab_len)

            freqo:freqObj = freq_list[label]
            freqo.addprior()
            
            dat_pos = data[0] 
            for n,obj_vocab_pos in enumerate(dat_pos):

                freq_of_obj = data[1+n]

                freqo.addy(obj_vocab_pos,freq_of_obj)
            
        self.freqlist:dict = freq_list
        self.samplelen = len(Y)

    def __predict(self,X):

        data_pos = X[0] #postions of the words used to decode the sparse representation
        scores = {}
        for label in self.freqlist:
            freqobject:freqObj = self.freqlist[label]
            probability_sum = math.log(freqobject.prior)-math.log(self.samplelen)
            for n,obj in enumerate(data_pos):
                
                freq_of_obj = X[1+n]
                probability_sum += freq_of_obj*math.log(freqobject.getProbability(obj))
            
            scores[label] = probability_sum
        return scores
        
    def predict_proba(self,X:list):
            scores = []
            for x in X:
                score = self.__predict(x)
                scores.append(score)

       
            return scores
    
    def predict(self,X:list):
        labels = []
        for x in X:
            score = self.__predict(x)
            labels.append(max(score, key=score.get))

            
            return labels
         
    def predict_proba_stream(self, X):
        for x in X:
            yield self.__predict(x)
    
                 