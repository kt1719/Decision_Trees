import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.core.fromnumeric import sort
from numpy.lib.arraysetops import unique
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors


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
    flatten_arr = np.ravel(training_dataset) #used to flatten the dataset into a 1d array
    room_num = flatten_arr[7:-1:8] #used to take each last column of the original matrix (room number)
    result = np.all(room_num == room_num[0])
    if result: #if all the samples have the same label
        node["value"] = flatten_arr[0]
        return (node, depth)
    else:
        node["depth"] = depth
        node["is_leaf"] = False
        node["attribute"], node["value"] = find_split(training_dataset)
        sorted_arr = training_dataset[np.argsort(training_dataset[:, node["attribute"]])]
        row_index = max(np.where(training_dataset == node["value"])[0])
        # val = node["value"]
        # attr = node["attribute"]
        # print(f"Depth is:{depth} the split is: {val} the attribute is: {attr}")
        node["left"], l_depth = decision_tree_learning(sorted_arr[0:row_index+1], depth+1)
        node["right"], r_depth = decision_tree_learning(sorted_arr[row_index+1:-1], depth+1)
        return (node, max(l_depth, r_depth))


        
def find_split(data): #chooses the attribute and the value that results in the highest information gain
    #   take in the dataset with all attributes and columns 
    #   return a tuple with the attribute and the number value

    if data.size == 0: #error handling (sanity check)
        raise Exception("Data in is nothing")

    entropy = float("inf") #entropy is initially max
    value = 0
    attribute = 0
    sorted_data = np.sort(data, axis=0)

    for i in range(0, len(sorted_data)-1):
        for unique_number in np.unique(data[i]):
            left_data = sorted_data[-1][sorted_data[i] <= unique_number]
            right_data = sorted_data[-1][sorted_data[i] > unique_number]
            remainder_num = remainder_calc(left_data, right_data)
            if (remainder_num < entropy):
                entropy = remainder_num
                value = unique_number
                attribute = i
    return (attribute, value)


# Some helper functions that should be defined
def entropy_calc(dataset):
    #   takes in 1-d array of room numbers
    #   returns the entropy value 
    entropy = 0
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
        segments += binary_tree_draw(levels - 1, xl, yl, width/2)
        segments += binary_tree_draw(levels - 1, xr, yr, width/2)
    return segments


data = np.loadtxt("clean_dataset.txt",)
tree, max_depth = decision_tree_learning(data)


segs = binary_tree_draw(max_depth, 0, 0, 15)

colors = [mcolors.to_rgba(c)
            for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
line_segments = LineCollection(segs, linewidths=1, colors=colors, linestyle='solid')



fig, ax = plt.subplots()
ax.set_xlim(-1, levels * depth_dist + 1)
ax.set_ylim(-1.5*width_dist, 1.5*width_dist)
ax.add_collection(line_segments)
plt.show()
