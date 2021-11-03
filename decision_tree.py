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
'''
width_dist = 10
depth_dist = 10
levels = 5


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
'''

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

def caculate_accuracy(confusion_matrix):
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


# to test your function (30 instances, 4 fold)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    Plot the confusion matrix
    '''
    print(title)
    print(cm)
    #classes = np.unique(actual).astype(int)
    classes = [1,2,3,4]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '2'
    thresh = cm.max() / 2.

    for i, j in ((i,j) for i in range(cm.shape[0]) for j in range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

#   return summed confusion matrix across k-folds and its accuracy
def k_fold_evaluation(data, k_fold=10):
    folds = k_fold_split(k_fold, data)
    big_conf = np.zeros((4,4))
    for (i,test_fold) in enumerate(folds):
        training_folds_combined = np.concatenate(folds[:i]+folds[i+1:])
        tree_test, max_depth_test = decision_tree_learning(training_folds_combined)
        conf_matrix = create_confusion_matrix(test_fold, tree_test)
        big_conf += conf_matrix
    return big_conf, caculate_accuracy(big_conf)

#return accuracy for a single test set
def evaluate(test_db, trained_tree):
    conf_matrix = create_confusion_matrix(test_db, trained_tree)
    return caculate_accuracy(conf_matrix)

def nested_cross_validation(data, k_fold=10):
#inout data and number of folds
#output big average of confusion matrices
#(1) For one training set -> 9 validation sets which gives 9 prune trees
#then average these into one confusion Matrix
#(2) After going through 9 other training sets which each have 9 different validation segments
#for each model do an averaged confusion matrix and then do a last average of the 10 different models
    folds = k_fold_split(k_fold, data) # 10 folds with 200 elements in each fold
    main_cm = np.zeros((4,4))
    for (i,test_fold) in enumerate(folds):
        outer_test_cm = np.zeros((4,4))
        remaining_folds = np.concatenate(folds[:i]+folds[i+1:])
       # validation_folds = k_fold_split(k_fold, test_fold) #10 validation folds with
        sum_norm_cm = np.zeros((4,4))
        for (j, validation_fold) in enumerate(remaining_folds):
            training_folds = np.concatenate(remaining_folds[:j]+remaining_folds[j+1:])
            tree, _ = decision_tree_learning(training_folds)
            #prunning of the tree using validation
            #compare accuracies of the 9 generated trees created, create an average confusion matrix using the test fold, keep track of the best tree
            conf_matrix = create_confusion_matrix(validation_fold, tree_test)
            norm_cm = conf_matrix / np.sum(conf_matrix, axis = 1)
            sum_norm_cm += norm_cm
            outer_test_cm += (sum_norm_cm / 9) # 1 test set and 9 different validation set big matrice
        main_cm += outer_test_cm

    big_norm_conf= main_cm / np.sum(main_cm, 1)
    return big_norm_conf / 10

#
'''
tests_folds = {}
big_conf = np.zeros((4,4))
for (i,test_fold) in enumerate(folds):
    training_folds_combined = np.concatenate(folds[:i]+folds[i+1:])
    tree_test, max_depth_test = decision_tree_learning(training_folds_combined)
    conf_matrix = create_confusion_matrix(test_fold, tree_test)
    tests_folds[i] = (conf_matrix, caculate_accuracy(conf_matrix))
    big_conf += conf_matrix
print()
print()
print("All folds: ")
print(tests_folds)
print()
print(big_conf)
print(caculate_accuracy(big_conf))
'''
data = np.loadtxt("clean_dataset.txt")
print(nested_cross_validation(data))

# x = np.array([-70, -50, -50, -50, -60, -60, -60, 2])
# temp = tree_test
# while not(temp["is_leaf"]):
#     t = temp["attribute"]
#     v = temp["value"]
#     print(f"attribute is: {t} value is: {v}")
#     if  x[int(temp["attribute_index"])] < temp["value"] :
#         temp = temp["left"]
#     else:
#         temp = temp["right"]
#     # store actual and predicted label
# gold_label = x[7]
# predicted_label = temp["value"]
# v = temp["value"]
# print(f"gold label is {gold_label}, predicted_label is: {predicted_label}")
