from numpy.random import default_rng
from decision_tree import *

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
        # Update confusion matrix
        confusion_matrix[int(predicted_label-1), int(gold_label-1)]+=1
        
    return confusion_matrix

#   returns an array of k folds
def k_fold_split(n_splits, data, random_generator=default_rng()):
    # generate a random permutation of data rows
    shuffled_indices = random_generator.permutation(data)
    # split shuffled indices into almost equal sized splits
    splits = np.array_split(shuffled_indices, n_splits)
    return splits
    
#   return accuracy for a single test set
def evaluate(test_db, trained_tree):
    conf_matrix = create_confusion_matrix(test_db, trained_tree)
    return calculate_accuracy(conf_matrix)

#   helper function for evaluate
def calculate_accuracy(confusion_matrix):
    #sum diagonals
    correct_predictions = np.trace(confusion_matrix)
    #sum all matrix entries
    total_predictions = np.sum(confusion_matrix)
    return correct_predictions/total_predictions 

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