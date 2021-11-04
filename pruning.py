from evaluation import *
from decision_tree import decision_tree_learning

def prune(node, validation_set, root_node):
    left = node["left"]
    right = node["right"]
    if node["left"]["is_leaf"] == False:
        prune(node["left"], validation_set, root_node)
    if node["right"]["is_leaf"] == False:
        prune(node["right"], validation_set, root_node)

    if left["is_leaf"] == True and right["is_leaf"] == True:
        #evaluate tree beforehand
        old_node = [int(node["attribute"]), node["value"], node["left"], node["right"], node["is_leaf"], node["depth"]]

        old_accuracy = evaluate(validation_set, root_node)

        #make a pruned version of the tree by simplifying the nodes
        length_node_l = node["left"]["attribute"]
        length_node_r = node["right"]["attribute"] #attribute means how many of the rooms are 

        newnode = []
        if length_node_l > length_node_r:
            newnode = [int(node["left"]["attribute"]), node["left"]["value"], node["left"]["left"], node["left"]["right"], node["left"]["is_leaf"], node["left"]["depth"]]
        elif length_node_r > length_node_l:
            newnode = [int(node["right"]["attribute"]), node["right"]["value"], node["right"]["left"], node["right"]["right"], node["right"]["is_leaf"], node["right"]["depth"]]
        
        if length_node_l != length_node_r:
            node["attribute"] = newnode[0]
            node["value"] = newnode[1]
            node["left"] = newnode[2]
            node["right"] = newnode[3]
            node["is_leaf"] = newnode[4]
            node["depth"] = newnode[5]
        #evaluate tree after
        new_accuracy = evaluate(validation_set, root_node)
        if new_accuracy < old_accuracy:
            #change the node back to what it was before
            node["attribute"] = old_node[0]
            node["value"] = old_node[1]
            node["left"] = old_node[2]
            node["right"] = old_node[3]
            node["is_leaf"] = old_node[4]
            node["depth"] = old_node[5]
            
    #return pruned version if evaluation is better or the same
    return None

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
        # print("sum_norm_cm: ")
        # print(sum_norm_cm / (k_fold-1))
        # print(" ")
        # print("cm_per_test_set : ")
        # print(cm_per_test_set)
        # print(" ")
        # print(" ")
    #print("Final Confusion Matrix Accuracy:")
    return cm_per_test_set

def pruning_and_evaluation(data):
    print("-Performance Metrics after pruning-" )
    pruned_confusion_matrix = nested_cross_validation(data)
    print("Confusion matrix:" )
    print(pruned_confusion_matrix)
    print("Accuracy: " + str(calculate_accuracy(pruned_confusion_matrix)))
    recall_list = calculate_recall(pruned_confusion_matrix)
    percision_list = calculate_percision(pruned_confusion_matrix)
    f1_list = calculate_f1(pruned_confusion_matrix)
    print("Metrics per class label-")
    for room, (recall, percision, f1) in enumerate(zip(recall_list, percision_list, f1_list)):
        print('Room '+str(room+1)+":")
        print("    Recall: " + str(recall))
        print("    Precision: " + str(percision))
        print("    F1: " + str(f1))