def word_lower_helper(sep):
    def helper(w):
        return w.lower().split(sep)
    
    return helper

class Vocabgen():
    
    def __init__(self):
        pass

    def __buildvocab(self,X:list,sep=' ',helper_function=None):
        '''
        loop each of the sentence and split words and return the unique words
        '''
        if helper_function == None:
            helper_function = word_lower_helper(sep)

        vocab = []
        vc = {}

        for i in X:
            objects = helper_function(i)

            for obj in objects:
                try:
                    k = vc[obj]
                except KeyError:
                    vocab.append(obj)
                    vc[obj] = {}
                    
        
        return vocab
    
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
