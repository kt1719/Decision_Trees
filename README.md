# Machine Learning Coursework : Decision trees <br />

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

```


### Intermediate Functions
```python
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
```
