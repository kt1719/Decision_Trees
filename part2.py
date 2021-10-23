import numpy as np
import matplotlib
import math

# Node dict should contain left, right, leaf boolean, and i guess conditions???
node = {"attribute": None,
        "value":
        "left" : 
        "right" :
        "is_leaf" :

}

        
def find_split(data):
    #   take in the dataset with all attributes and columns 
    #   return a dictionary 
    x = np.unique(data)
    return
    


# Some helper functions that should be defined
def entropy_calc(dataset):
    #   takes in 1-d array of room numbers
    #   returns the entropy value 
    x = np.unique(dataset)
    for i in x:
        prob = np.sum(dataset == i) / len(dataset)
        entropy =+ -1 * prob * math.log(prob, 2)
    return entropy


def gain_calc(s_all, s_left, s_right):
    return entropy_calc(s_all) - remainder_calc(s_left, s_right)


    

def remainder_calc(s_left, s_right):
    return 


