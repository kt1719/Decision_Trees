# Machine Learning Coursework : Decision trees <br />

## Contributors
* [Ian Ren](https://github.com/ianzren) : [ian.ren19@imperial.ac.uk]
* [Khayle Torres](https://github.com/kt1719) : [khayle.torres19@imperial.ac.uk]
* [Salman Dhaif](https://github.com/sdoif) : [salman.dhaif19@imperial.ac.uk]
* [Yuna Valade](https://github.com/yv19) : [yuna.valade19@imperial.ac.uk]

## Instructions
1. Run "python3 main.py" in terminal to see evaluation metrics for clean and noisy datasets.
2. Uncomment lines under Section 2 to see the visualisation of the these trees.
3. Uncomment sections below to see the accuracy as well as the tree visualisation before and after pruning.
4. To test secret test data, uncomment final section, line 48 onwards, and comment previous sections. Add path to datasourse in secret_data and run again.
 <br />


## Main Functions
```python
#Function used to return a tuple of a trained tree and a max depth
def decision_tree_learning(training_dataset, depth=0):

#Function used to plot the tree
def plot_tree(tree, max_depth, title):

#Function used to compute 
def cross_evaluation(data, k_fold=10):

#Function used to return a single (evaluate function is used in prune and also prune_test)
def evaluate(test_db, trained_tree):

#Function used to prune the instance of the tree (Returns nothing as it prunes the tree directly)
#This is done to avoid the alternative route of deepcopy time and memory complexity
#Downside is once a tree is pruned it cannot be unpruned
def prune(node, validation_set, root_node):

#Function to  to create a tree and calculate its corresponding normalised average confusion matrix using nested cross validation (Used in pruning_and_evaluation)
def nested_cross_validation(data, k_fold=10):

#Function used to calculate a confusion matrix's corresponding statistics (F1, accuracy, etc..)
def pruning_and_evaluation(data, k_fold=10):

```


### Intermediate Functions
```python
### HELPER FUNCTIONS FOR "decision_tree_learning" & "plot_tree" ###

#Function used to the find a split in the dataset that gives the highest information gain
def find_split(data):

#Function used to calcualate entropy
def entropy_calc(dataset):

#Function used to calculate gain
def gain_calc(s_all, s_left, s_right):

#Function the calculate the remainder
def remainder_calc(s_left, s_right):

#Function used to check if the node is a leaf
def check_leaf(node):

############################################################################################################

### HELPER FUNCTIONS FOR "evaluate" & "cross_evaluation" & "nested_cross_validation" ###

#Function for looping through different splits and feeding into "create_confusion_matrix"
#The function would then normalize each matrix outputted, sum it, and find the average
def k_fold_confusion_matrix_calc(data, k_fold=10):

#Function used to compute confusion matrix by testing the tree with the test_db dataset
def create_confusion_matrix(test_db, trained_tree):

#Function used to randomise and split an array into k folds
def k_fold_split(k_folds, data, random_generator=default_rng()):

#Function used to calculate the accuracy for each class label
def calculate_accuracy(confusion_matrix):

#Function used to calculate recall for each class label
def calculate_recall(confusion_matrix):

#Function used to calculate precision for each class label
def calculate_precision(confusion_matrix):

#Function used to calculate F1 for each class label
def calculate_f1(confusion_matrix):

############################################################################################################

### HELPER FUNCTION FOR "plot tree" ###

#Function used to compute x,y coordinates for the lines and annotations in order to use for plotting purposes
def binary_tree_draw(tree, x, y, width=5):

############################################################################################################

### FOR UNDERSTANDING AND TESTING PURPOSES ###

#Macrofunction that plots the graphs before and after pruning and also prints it's number of leaves + max depth
def prune_test(data):

#MISC function that computes the number of leaves a tree has
def countleaves(node):

#MISC function that computes a corresponding tree's maximum depth
def maxdepth(node):
```
