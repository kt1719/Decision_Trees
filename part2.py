import numpy as np
import matplotlib
import math
from numpy.core.fromnumeric import sort
from numpy.lib.arraysetops import unique

# Node dict should contain left, right, leaf boolean, and i guess conditions???
node = {"value": None,
        "left" : None,
        "right" : None,
        "is_leaf" : False
}

        
def find_split(data):
    #   take in the dataset with all attributes and columns 
    #   return a dictionary
    node = {"value": 0,
        "left" : None,
        "right" : None,
        "is_leaf" : True
    }

    if data.size == 0:
        return node

    entropy = float("inf")
    node["is_leaf"] = False

    unique_values = np.unique(data, axis=0)
    sorted_indecies= np.sort(data, axis=0)
    index_i = 0
    index_j = 0
    for (ind_i,i) in enumerate(unique_values):
        left_data = np.empty([])
        right_data = np.empty([])
        temp_split_index = ind_i
        for (ind_j,j) in unique_values[ind_i]:
            temp_split_num = ind_j
            np.insert(right_data, data[sorted_indecies>j, i])
            np.insert(left_data, data[sorted_indecies<=j, i])
            remainder_num = remainder_calc(left_data, right_data)
            if (remainder_num < entropy):
                entropy = remainder_num
                node["value"] = j
                index_i = ind_i
                index_j = ind_j
    
    return node

def find_split(data):
    '''
       take in the dataset with all attributes and columns 
       return a dictionary with a structure 

       node = {"attribute": None, "value": None, "left" : None, "right" : None, "is_leaf" : None}
    1 - First we need to 
    2 - 
    3 - 

    '''



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
    total_cardinality = len(s_left)+len(s_right)
    h_left = entropy_calc(s_left)
    w_left = len(s_left)/total_cardinality
    h_right = entropy_calc(s_right)
    w_right = len(s_right)/total_cardinality
    return (w_left*h_left)+(w_right*h_right)


