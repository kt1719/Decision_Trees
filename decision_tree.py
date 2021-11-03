import numpy as np
import math

def decision_tree_learning(training_dataset, depth=0):
    node = {
        "attribute": 0,
        "value": 0,
        "left" : None,
        "right" : None,
        "is_leaf" : True,
        "depth" : depth
    }
    arr = training_dataset[:,-1]
    result = np.max(arr) == np.min(arr)
    if result: #if all the samples have the same label
        node["attribute"] = len(arr) #since attribute has no meaning to a leaf this stores the number of cases there is for the room 
                                           #which will be used for pruning (Getting the most common room)
        node["value"] = arr[0]
        return (node, depth)
    else:
        sorted_arr = np.empty([])
        node["depth"] = depth
        node["is_leaf"] = False
        node["attribute"], node["value"], row_index = find_split(training_dataset)

        sorted_arr = training_dataset[np.argsort(training_dataset[:, node["attribute"]])]
        val = node["value"]
        attr = node["attribute"]
        node["right"], r_depth = decision_tree_learning(sorted_arr[:row_index+1], depth+1)
        node["left"], l_depth = decision_tree_learning(sorted_arr[row_index+1:], depth+1)
        return (node, max(l_depth, r_depth))

#   chooses the attribute and the value that results in the highest information gain
#   take in the dataset with all attributes and columns 
#   return a tuple with the attribute and the number value
def find_split(data): #chooses the attribute and the value that results in the highest information gain
    #   take in the dataset with all attributes and columns 
    #   return a tuple with the attribute and the number value

    if data.size == 0: #error handling (sanity check)
        raise Exception("Data in is nothing")

    gain = 0 # gain is initially 0
    value = 0
    attribute = 0
    row_index = 0
    for i in range(data.shape[1]-1):
        sorted_data = data[np.argsort(data[:, i])]
        b = sorted_data[:,i] #take all the unique columns
        room_num = sorted_data[:,-1]
        for unique_num in np.unique(b):
            left_data = room_num[b <= unique_num]
            right_data = room_num[b > unique_num]
            information_gain = gain_calc(sorted_data[:,-1],left_data, right_data)
            if (information_gain > gain):
                gain = information_gain
                value = unique_num
                attribute = i
                arr = np.where(b == unique_num)
                row_index = np.max(arr)
    return (attribute, value, row_index)


# Helper functions
def entropy_calc(dataset):
    #   takes in 1-d array of room numbers
    #   returns the entropy value
    entropy = 0
    x = np.unique(dataset)
    for i in x:
        prob = np.sum(dataset == i) / len(dataset)
        entropy -= prob * math.log(prob, 2)
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

def check_leaf(node):
    if node["left"] == None and node["right"] == None:
        return True
    return False
