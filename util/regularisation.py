'''
A  Utility to add regularisation on the models.This can be reused in linear,logistic regression and Neural Network(in future)
'''

class Regularisation:
    '''
    Static class for holding all regularisation functions
    '''
    def L2(regularisation_power):
        def exec(W:int):
            return 2*W*regularisation_power
        
        return exec
    
    def L1(regularisation_power):
        def exec(W:int):

            sgn = 1
            if W < 0:
                sgn = -1
            elif W == 0:
                sgn = 0

            return regularisation_power*sgn
        
        return exec