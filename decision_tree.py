import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import math
from numpy.core.fromnumeric import searchsorted, sort
from numpy.lib.arraysetops import unique
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from numpy.lib.index_tricks import s_
from numpy.random import default_rng 


# Node dict should contain left, right, leaf boolean, and i guess conditions???
node = {
        "attribute": None,
        "value": None,
        "left" : None,
        "right" : None,
        "is_leaf" : False,
        "depth" : None
}

def decision_tree_learning(training_dataset, depth=0):
    node = {
        "attribute": 0,
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
        node["right"], r_depth = decision_tree_learning(sorted_arr[:row_index+1], depth+1)
        node["left"], l_depth = decision_tree_learning(sorted_arr[row_index+1:], depth+1)
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
def binary_tree_draw(tree, x, y, width): #from stackoverflow
    attr = tree["attribute"]
    val = tree["value"]
    segments = []
    if tree["is_leaf"]:
        plt.annotate(f"Room is: {val}", (x,y), ha="center", size=8, bbox = dict(boxstyle="round", pad=0.3, lw = 0.5, fc = "white", ec="b"))
    else:
        yl = y - depth_dist
        xl = x - width 
        xr = x + width 
        yr = y - depth_dist
        segments.append([[x,y], [xl,yl]])
        segments.append([[x,y], [xr,yr]])
        plt.annotate(f"x{attr} <= {val}", (x,y), ha="center", size=8, bbox = dict(boxstyle="round", pad=0.3, lw = 0.5, fc = "white", ec="b"))
    if tree["left"] != None:
        segments += binary_tree_draw(tree["left"], xl, yl, width/2)
    if tree["right"] != None:
        segments += binary_tree_draw(tree["right"], xr, yr, width/2)
    return segments

def check_leaf(node):
    if node["left"] == None and node["right"] == None:
        return True
    return False

n_folds = 10

def evaluate_tree(data):
    total_error = 0
    for (train_indices, test_indices) in k_fold_split(n_folds, data):
        trained_tree = decision_tree_learning(train_indices)
        total_error += evaluate(test_indices, trained_tree)    
    return total_error / n_folds

# Find accuracy for a single decision tree
def create_confusion_matrix(test_db, trained_tree):
    # Initizalize confusion matrix
    confusion_matrix = np.zeros([4,4], dtype=int)
    
    # Traverse the trained_tree and validate if its the samte as the last column in test_db
    for row in test_db:
        temp = trained_tree
        while not(temp["is_leaf"]):
            attribute = temp["attribute"]
            value = temp["value"]
            if  row[int(attribute)] <= value :
                temp = temp["right"]
            else:
                temp = temp["left"]
        # store actual and predicted label
        gold_label = row[7]
        predicted_label = temp["value"]
        v = temp["value"]
        x = row[int(temp["attribute"])]
        # Update confusion matrix
        confusion_matrix[int(predicted_label-1), int(gold_label-1)]+=1
        
    return confusion_matrix
    
def calculate_accuracy(confusion_matrix):
    #sum diagonals
    correct_predictions = np.trace(confusion_matrix)
    #sum all matrix entries
    total_predictions = np.sum(confusion_matrix)
    return correct_predictions/total_predictions 

def k_fold_split(n_splits, data, random_generator=default_rng()):
    # generate a random permutation of data rows
    shuffled_indices = random_generator.permutation(data)

    # split shuffled indices into almost equal sized splits
    splits = np.array_split(shuffled_indices, n_splits)

    return splits

# Splits data into K folds
def train_k_fold_split(n_folds, data):
    split_indices = np.array_split(data, n_folds)
    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]
        # combine remaining splits as train
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])
        folds.append([train_indices, test_indices])
    return folds

#   function takes in entire dataset, splits it into k folds (10 by default)
#   take turns using each fold as the dataset and the remaining to train the tree
#   return the average confusion matrix by:
#   compute confusion matrix for each training set, normalize it, then sum them all / 10
def k_fold_confusion_matrix_calc(data, k_fold=10):
    folds = k_fold_split(k_fold, data)
    sum_norm_confusion = np.zeros((4,4))
    for (i,test_fold) in enumerate(folds):
        training_folds_combined = np.concatenate(folds[:i]+folds[i+1:])
        tree_test, max_depth_test = decision_tree_learning(training_folds_combined)
        confusion_matrix = create_confusion_matrix(test_fold, tree_test)
        norm_confusion = confusion_matrix / np.sum(confusion_matrix, axis = 1) 
        sum_norm_confusion += norm_confusion
    return sum_norm_confusion/10
    
#   return accuracy for a single test set
def evaluate(test_db, trained_tree):
    conf_matrix = create_confusion_matrix(test_db, trained_tree)
    return calculate_accuracy(conf_matrix)

#   return recall rate for each class label
def calculate_recall(confusion_matrix):
    recall_list = []
    for i, row in enumerate(confusion_matrix):
        recall_list.append(row[i]/np.sum(row))
    return recall_list

#   return percision rate for each class label
def calculate_percision(confusion_matrix):
    percision_list = []
    #transpose the matrix to acess column elements
    for i, column in enumerate(confusion_matrix.T):
        percision_list.append(column[i]/np.sum(column))
    return percision_list

#   return F1 measure for each class label
def calculate_f1(confusion_matrix):
    #   calculate recall and percision rates for each class label
    recall_list = calculate_recall(confusion_matrix)
    percision_list = calculate_percision(confusion_matrix)
    f1_measures = []
    #   zip rates for each class label together and calculate f1 measure
    for recall, percision in zip(recall_list,percision_list):
        f1_measures.append((2*recall*percision)/(recall+percision))
    return f1_measures

clean_data = np.loadtxt("clean_dataset.txt")
noisy_data = np.loadtxt("noisy_dataset.txt")

print()

print("Clean Data Statistics: ")
average_confusion= k_fold_confusion_matrix_calc(clean_data)
print(average_confusion)
print(calculate_accuracy(average_confusion))
print(calculate_recall(average_confusion))
print(calculate_percision(average_confusion))
print(calculate_f1(average_confusion))

print()

print("Noisy Data Statistics: ")
average_confusion = k_fold_confusion_matrix_calc(noisy_data)
print(average_confusion)
print(calculate_accuracy(average_confusion))
print(calculate_recall(average_confusion))
print(calculate_percision(average_confusion))
print(calculate_f1(average_confusion))

print()
