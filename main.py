from evaluation import *
from plot_decision_tree import plot_prune_tree, plot_tree

def cross_valid_evaluation(data):
    average_confusion= k_fold_confusion_matrix_calc(data)
    print("Confusion matrix:" )
    print(average_confusion)
    print("Accuracy: " + str(calculate_accuracy(average_confusion)))
    print("Recall: " + str(calculate_recall(average_confusion)))
    print("Precision: " + str(calculate_percision(average_confusion)))
    print("f1: " + str(calculate_f1(average_confusion)))


clean_data = np.loadtxt("wifi_db/clean_dataset.txt")
noisy_data = np.loadtxt("wifi_db/noisy_dataset.txt")

# Clean dataset cross valid evaluation
print("Clean Data Statistics: ")
cross_valid_evaluation(clean_data)
# UNCOMMENT LINE BELOW TO SEE TREE
plot_tree(clean_data)

print()

# Noisy dataset cross valid evaluation
print("Noisy Data Statistics: ")
cross_valid_evaluation(noisy_data)
# UNCOMMENT LINE BELOW TO SEE TREE
# plot_tree(noisy_data)

# print()

# Draw pruned tree for clean and noisy data

# print("Pruned trees and before and after accuracies for clean_dataset")
# plot_prune_tree(clean_data)

# print()

# print("Pruned trees and before and after accuracies for clean_dataset")
# plot_prune_tree(noisy_data)



