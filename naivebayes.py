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

class NaiveBayes():
    
    def __init__(self):
        pass

    