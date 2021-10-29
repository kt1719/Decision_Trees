import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import math
from numpy.core.fromnumeric import searchsorted, sort
from numpy.lib.arraysetops import unique
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from numpy.lib.index_tricks import s_


# Node dict should contain left, right, leaf boolean, and i guess conditions???
node = {
        "attribute_index": None,
        "value": None,
        "left" : None,
        "right" : None,
        "is_leaf" : False,
        "depth" : None
}

def decision_tree_learning(training_dataset, depth=0):
    node = {
        "attribute_index": None,
        "value": 0,
        "left" : None,
        "right" : None,
        "is_leaf" : True,
        "depth" : None
    }
    arr = training_dataset[:,-1]
    result = np.max(arr) == np.min(arr)
    if result: #if all the samples have the same label
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
        print(f"Depth is:{depth} the split is: {val} the attribute is: {attr} split row is: {row_index} shape is: {sorted_arr.shape} size of left is {sorted_arr[:row_index+1].shape} size of right is {sorted_arr[row_index+1:].shape}")
        node["left"], r_depth = decision_tree_learning(sorted_arr[:row_index+1], depth+1)
        node["right"], l_depth = decision_tree_learning(sorted_arr[row_index+1:], depth+1)
        return (node, max(l_depth, r_depth))


        
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

    # for i in range(0, len(sorted_data)-1):
    #     for unique_number in np.unique(sorted_data[i]):
    #         left_data = sorted_data[-1][sorted_data[i] <= unique_number]
    #         right_data = sorted_data[-1][sorted_data[i] > unique_number]
    #         information_gain = gain_calc(sorted_data[-1],left_data, right_data)
    #         if (information_gain > gain):
    #             gain = information_gain
    #             value = unique_number
    #             attribute = i
    # return (attribute, value)


# Some helper functions that should be defined
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

width_dist = 10
depth_dist = 10
levels = 5 

#stackoverflow.com/questions/59028711/plotting-a-binary-tree-in-matplotlib
def binary_tree_draw(levels, x, y, width): #from stackoverflow
    segments = []
    yl = y - depth_dist
    xl = x - width / 2
    xr = x + width/2
    yr = y - depth_dist
    segments.append([[x,y], [xl,yl]])
    segments.append([[x,y], [xr,yr]])
    if levels > 1:
        segments += binary_tree_draw(levels - 1, xl, yl, width*1.5)
        segments += binary_tree_draw(levels - 1, xr, yr, width*1.5)
    return segments


data = np.loadtxt("clean_dataset.txt",)
tree, max_depth = decision_tree_learning(data)


print(max_depth)
segs = binary_tree_draw(max_depth, 0, 0, 5)

colors = [mcolors.to_rgba(c)
            for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
line_segments = LineCollection(segs, linewidths=1, colors=colors, linestyle='solid')



fig, ax = plt.subplots()
ax.set_xlim(-1, levels * depth_dist + 1)
ax.set_ylim(-1.5*width_dist, 1.5*width_dist)
ax.add_collection(line_segments)
plt.show()

# a = np.array([2,2,2,2,2,1,1,1,1,1])
# b = np.array([2,2,2,2,2,1,1,1,1])
# print(remainder_calc(a,b))




# a = np.array([[-51,-59,-51,-48,-67,-79,-79, 3],
# [-46,-59,-54,-52,-67,-77,-83, 2],
# [-46,-59,-55,-49,-69,-74,-84, 2],
# [-45,-59,-54,-51,-69,-77,-85, 2],
# [-45,-57,-55,-48,-67,-77,-84, 2]])

# attr, val, row = find_split(a)
# print(attr, "attribute")
# print(val, "value")
# print(row, "row")

n_folds = 10

def evaluate_tree(data):
    total_error = 0
    for (train_indices, test_indices) in k_fold_split(n_folds, data):
        trained_tree = decision_tree_learning(train_indices)
        total_error += evaluate(test_indices, trained_tree)    
    return total_error / n_folds

def evaluate(test_db, trained_tree):
    # Traverse the trained_tree and validate if
    # its the same as the last collumn in test_db
    return 0

# Splits data into K folds
def k_fold_split(n_folds, data):

    split_indices = np.array_split(data, n_folds)

    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]

        # combine remaining splits as train
        # this solution is fancy and worked for me
        # feel free to use a more verbose solution that's more readable
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        folds.append([train_indices, test_indices])

    return folds


# to test your function (30 instances, 4 fold)

