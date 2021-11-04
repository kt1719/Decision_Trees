from pruning import *
from plot_decision_tree import prune_test, plot_tree

###Secton 1###
clean_data = np.loadtxt("wifi_db/clean_dataset.txt")
noisy_data = np.loadtxt("wifi_db/noisy_dataset.txt")

############################################################################################################

###Section 2###
print("Clean Data Decision Tree: ")
plot_tree(decision_tree_learning(clean_data))
print("Noisy Data Decision Tree: ")
plot_tree(decision_tree_learning(noisy_data))

############################################################################################################

###Section 3###

# Clean dataset cross valid evaluation
print("Clean Data Statistics: ")
cross_evaluation(clean_data)
print()

### Noisy dataset cross valid evaluation ###
print("Noisy Data Statistics: ")
cross_evaluation(noisy_data)
print()

############################################################################################################

###Section 4###

# COMPARE BEFORE AND AFTER PRUNING (This is for visualization and stat purposes of the tree)
print("Pruned trees and before and after accuracies for clean_dataset")
prune_test(clean_data)
print()
pruning_and_evaluation(clean_data)
print()

print("Pruned trees and before and after accuracies for noisy_dataset")
prune_test(noisy_data)
print()
pruning_and_evaluation(noisy_data)
print()

############################################################################################################
'''
FOR SECRET TEST DATA
path_to_file = ""
secret_data = np.loadtxt(path_to_file)

print("Secret Data Decision Tree: ")
plot_tree(decision_tree_learning(secret_data))

print("Secret Data Statistics: ")
cross_evaluation(secret_data)
print()

print("Pruned trees and before and after accuracies for clean_dataset")
prune_test(secret_data)
print()
pruning_and_evaluation(secret_data)
print()
'''