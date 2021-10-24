import numpy as np
import matplotlib
import math
from numpy.core.fromnumeric import sort
from numpy.lib.arraysetops import unique

# Node dict should contain left, right, leaf boolean, and i guess conditions???
node = {
        "attribute_index": None,
        "value": None,
        "left" : None,
        "right" : None,
        "is_leaf" : False
}

def decision_tree_learning(training_dataset, depth):
    node = {
        "attribute_index": None,
        "value": 0,
        "left" : None,
        "right" : None,
        "is_leaf" : True
    }
    flatten_arr = np.ravel(training_dataset) #used to flatten the dataset into a 1d array
    room_num = flatten_arr[6:-1:7] #used to take each last column of the original matrix (room number)
    result = np.all(room_num == room_num[0])
    if result: #if all the samples have the same label
        node["value"] = flatten_arr[0]
        return node
    else:
        node["is_leaf"] = False
        [node["attribute"], node["value"]] = find_split(training_dataset)
        [node["left"], l_depth] = decision_tree_learning(training_dataset<=node["value"], depth+1)
        [node["right"], r_depth] = decision_tree_learning(training_dataset>node["value"], depth+1)
        return (node, max(l_depth, r_depth))


        
def find_split(data): #chooses the attribute and the value that results in the highest information gain
    #   take in the dataset with all attributes and columns 
    #   return a tuple with the attribute and the number value

    if data.size == 0: #error handling (sanity check)
        raise Exception("Data in is nothing")

    entropy = float("inf") #entropy is initially max
    value = 0
    attribute = 0
    sorted_data= np.sort(data, axis=0)

    for i in range(0, sorted_data.size()-1):
        for unique_number in np.unique(sorted_data[i]):
            left_data = sorted_data[-1][sorted_data <= unique_number]
            right_data = sorted_data[-1][sorted_data > unique_number]
            remainder_num = remainder_calc(left_data, right_data)
            if (remainder_num < entropy):
                entropy = remainder_num
                value = unique_number
                attribute = i
    return (value, attribute)


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
