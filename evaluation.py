import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import math
from numpy.core.fromnumeric import searchsorted, sort
from numpy.lib.arraysetops import unique
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from numpy.lib.index_tricks import s_
import decision_tree

# Splits data into K folds
def k_fold_split_old(data, n_folds):
    split_indices = np.array_split(data, n_folds)
    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]
        # combine remaining splits as train
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])
        folds.append([train_indices, test_indices])
    return folds

# Splits data into K folds. Returns indices not data!!!!!
def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)
    # split shuffled indices into almost equal sized splits
    folds = np.array_split(shuffled_indices, n_splits)
    return folds

# Returns array of predicted results
def get_predicted_result(test_db, trained_tree):
    results = []
    for row in range(0, test_db.shape[0] - 1):
        temp = trained_tree.copy()
        print(temp)
        while not(temp["is_leaf"]):
            if  temp["value"] < test_db[row, temp["attribute_index"]]:
                temp = temp["left"]
            else:
                temp = temp["right"]
        results.append(int(temp["value"]))
    return results


# TODO Add to Confusion Matrix
def confusion_matrix(y_gold, y_prediction, class_labels=None):

    # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)
    # for each correct class (row), 
    # compute how many instances are predicted for each class (columns)
    for (i, label) in enumerate(class_labels):
        # get predictions where the ground truth is the current class label
        indices = (y_gold == label)
        gold = y_gold[indices]
        predictions = y_prediction[indices]
        # quick way to get the counts per label
        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))
        # fill up the confusion matrix for the current row
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)
    return confusion

# TODO Cross Validation function
def cross_validation_split(data, n_folds=10):
    folds = k_fold_split(n_folds, data)
    confusions = []
    for (i, fold) in enumerate(folds):
        test_db = data[fold]
        training_folds_combined = np.concatenate(folds[:i]+folds[i+1:])
        print("THIS IS test_db")
        print(test_db)
        trained_tree, max_depth = decision_tree.decision_tree_learning(training_folds_combined)
        print("THIS IS training data")
        print(training_folds_combined)
        y_gold = test_db[:, -1]
        y_prediction = get_predicted_result(test_db, trained_tree)
        print(y_gold)
        print(y_prediction)
        confusions.append(confusion_matrix(y_gold, y_prediction))
    confusion = np.sum(confusion_matrix, 0)

    return confusion

print("IAN IAN IAN")
data = np.loadtxt("clean_dataset.txt")
cm = cross_validation_split(data)
plot_confusion_matrix(cm)



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