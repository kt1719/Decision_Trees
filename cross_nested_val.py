from decision_tree import *
from evaluation import *
from pruning import prune

def nested_cross_validation(data, k_fold=10):
#input data and number of folds
#output big average of confusion matrices
#(1) For one training set -> 9 validation sets which gives 9 prune trees
#then average these into one confusion Matrix
#(2) After going through 9 other training sets which each have 9 different validation segments
#for each model do an averaged confusion matrix and then do a last average of the 10 different models
    folds = k_fold_split(k_fold, data) # 10 folds with 200 elements in each fold
    big_norm_conf = np.zeros((4,4))
    cm_per_test_set = np.zeros((4,4))
    for (i,test_fold) in enumerate(folds):
        rest = np.concatenate(folds[:i]+folds[i+1:])
        remaining_folds = np.array_split(rest, k_fold-1)
       # validation_folds = k_fold_split(k_fold, test_fold) #10 validation folds with
        sum_norm_cm = np.zeros((4,4))
        for (j, validation_fold) in enumerate(remaining_folds):
            training_folds = np.concatenate(remaining_folds[:j]+remaining_folds[j+1:])
            tree, _ = decision_tree_learning(training_folds)
            #prunning of the tree using validation
            prune(tree, validation_fold, tree)
            #compare accuracies of the 9 generated trees created, create an average confusion matrix using the test fold, keep track of the best tree
            conf_matrix = create_confusion_matrix(test_fold, tree)
            norm_cm = conf_matrix / np.sum(conf_matrix, axis = 1)
            sum_norm_cm += norm_cm # 1 test set and 9 different validation set big matrice
        cm_per_test_set += (sum_norm_cm/ (k_fold-1))
    return cm_per_test_set

def pruning_and_evaluation(data, k_fold=10):
    print("-Performance Metrics after pruning-" )
    pruned_confusion_matrix = nested_cross_validation(data, k_fold)
    print("Confusion matrix:" )
    print(pruned_confusion_matrix)
    print("Accuracy: " + str(calculate_accuracy(pruned_confusion_matrix)))
    recall_list = calculate_recall(pruned_confusion_matrix)
    percision_list = calculate_precision(pruned_confusion_matrix)
    f1_list = calculate_f1(pruned_confusion_matrix)
    print("Metrics per class label-")
    for room, (recall, percision, f1) in enumerate(zip(recall_list, percision_list, f1_list)):
        print('Room '+str(room+1)+":")
        print("    Recall: " + str(recall))
        print("    Precision: " + str(percision))
        print("    F1: " + str(f1))