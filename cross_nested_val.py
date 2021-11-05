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
        print("sum_norm_cm: ")
        print(sum_norm_cm / (k_fold-1))
        print(" ")
        print("cm_per_test_set : ")
        print(cm_per_test_set)
        print(" ")
        print(" ")
    print("Final Confusion Matrix Accuracy:")
    return calculate_accuracy(cm_per_test_set / k_fold)
