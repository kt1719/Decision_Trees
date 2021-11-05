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
        #evaluate tree after
        new_accuracy = evaluate(validation_set, root_node)
        if new_accuracy < old_accuracy:
            #change the node back to what it was before
            node["attribute"] = old_node[0]
            node["value"] = old_node[1]
            node["left"] = old_node[2]
            node["right"] = old_node[3]
            node["is_leaf"] = old_node[4]
            
    #return pruned version if evaluation is better or the same
    return None