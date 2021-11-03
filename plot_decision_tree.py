import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from decision_tree import decision_tree_learning
from evaluation import *
from pruning import prune

#stackoverflow.com/questions/59028711/plotting-a-binary-tree-in-matplotlib

width_dist = 10
depth_dist = 10
levels = 5 

def binary_tree_draw(tree, x, y, width):
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

def plot_tree(data):
    tree, max_depth = decision_tree_learning(data)
    fig2, ax2 = plt.subplots()
    segs2 = binary_tree_draw(tree, 0, 0, 5)
    colors = [mcolors.to_rgba(c)
                for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    line_segments = LineCollection(segs2, linewidths=1, colors=colors, linestyle='solid')

    ax2.set_xlim(-width_dist, width_dist)
    ax2.set_ylim(-(max_depth +1)* depth_dist -5 , 5)
    ax2.add_collection(line_segments)
    plt.show()

# TODO Turn this function into prune_test() and call plot_tree() instead of repeating code
def plot_prune_tree(data):
    folds = k_fold_split(10, data)
    training_folds = np.concatenate(folds[2:])
    validation_folds = folds[1]
    testing_folds = folds[0]
    tree, max_depth = decision_tree_learning(training_folds)

    fig2, ax2 = plt.subplots()
    segs2 = binary_tree_draw(tree, 0, 0, 5)
    colors = [mcolors.to_rgba(c)
                for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    line_segments = LineCollection(segs2, linewidths=1, colors=colors, linestyle='solid')

    ax2.set_xlim(-width_dist, width_dist)
    ax2.set_ylim(-(max_depth +1)* depth_dist -5 , 5)
    ax2.add_collection(line_segments)

    print("Old accuracy: " + str(evaluate(testing_folds, tree)))

    prune(tree, validation_folds, tree)

    print("New accuracy: " + str(evaluate(testing_folds,tree)))

    fig, ax = plt.subplots()
    segs = binary_tree_draw(tree, 0, 0, 5)

    colors = [mcolors.to_rgba(c)
                for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    line_segments = LineCollection(segs, linewidths=1, colors=colors, linestyle='solid')

    ax.set_xlim(-width_dist, width_dist)
    ax.set_ylim(-(max_depth +1)* depth_dist -5 , 5)
    ax.add_collection(line_segments)
    plt.show()