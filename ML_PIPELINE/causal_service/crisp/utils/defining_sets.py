import numpy as np
from treelib import Tree

"""
Python re-implentation of https://github.com/christinaheinze/nonlinearICP-and-CondIndTests/blob/master/nonlinearICP/R/computeDefiningSets.R
"""


def defining_sets(accepted_subsets):
    defining_sets_list = []
    accepted_subsets = list(accepted_subsets)
    # Remove all sets that are supersets of the minLength sets
    accepted_subsets = get_minimal_subsets(accepted_subsets)

    if len(accepted_subsets):
        # add all singletons
        singletons = []
        for sets in accepted_subsets:
            if len(sets) == 1:
                singletons.append(sets)

        tree_list = []
        root_ids = []
        if len(singletons) > 0:
            for i in range(len(singletons)):
                if i == 0:
                    sing_tree = Tree()
                    sing_tree.create_node(singletons[0][0], singletons[0][0])
                    root_ids.append(singletons[0][0])
                    tree_list.append(sing_tree)
                else:
                    tree_list[0].create_node(singletons[i][0], singletons[i][0], parent=singletons[i - 1][0])
                # Remove singleton list from accepted_subset
                accepted_subsets = [l for l in accepted_subsets if singletons[i][0] not in l]

        if len(accepted_subsets) > 0:
            flat_list = [item for sublist in accepted_subsets for item in sublist]
            uniq_vals, counts = np.unique(flat_list, return_counts=True)
            if len(tree_list) == 0:
                for i in range(len(uniq_vals)):
                    tree = Tree()
                    tree.create_node(uniq_vals[i], uniq_vals[i])
                    root_ids.append(uniq_vals[i])
                    tree_list.append(tree)
            else:
                for i, tree in enumerate(tree_list):
                    # find leaf
                    leaf = tree.leaves(root_ids[i])[0].identifier
                    for val in uniq_vals:
                        tree.create_node(val, val, parent=leaf)

            print('Defining Set Trees: ', len(tree_list))
            for i in range(len(tree_list)):
                print('Growing tree ', i)
                cont = True
                cc = 0
                while cont and cc < 5:
                    leaves = tree_list[i].leaves(root_ids[i])
                    # list of node from leaf to root for each leaf
                    leaf_node_paths = [get_path_to_root_for(tree_list[i], leaf) for leaf in leaves]
                    continue_any = np.ones([len(leaf_node_paths)])
                    for j, leaf in enumerate(leaves):
                        # remove subsets for which leaf is present in subset
                        asReduced = []
                        for se in accepted_subsets:
                            remove = False
                            for tag in leaf_node_paths[j]:
                                if tag in se:
                                    remove = True
                            if not remove:
                                asReduced.append(se)

                        if sum([len(l) for l in asReduced]) == 0:
                            continue_any[j] = 0
                        else:
                            tree_list[i] = build_trees(asReduced, tree_list[i], leaf)
                    cont = continue_any.any()
                    cc += 1

        # prune
        print('Pruning defining set trees')
        sets = []
        for i, tree in enumerate(tree_list):
            # convert tree to table with each row as a node, columns 'pathString', 'isLeaf'
            all_leaves = tree.leaves(root_ids[i])
            for node in all_leaves:
                lis = list(str(get_path_string(tree, node)).split('/'))
                lis.sort()
                sets.append(lis)

        sets = get_minimal_subsets(sets)

        return sets
    else:
        return defining_sets_list


def build_trees(as_reduced, tree, add_to):
    flat_list = [item for sublist in as_reduced for item in sublist]
    uniq_vals, counts = np.unique(flat_list, return_counts=True)
    for val in uniq_vals:
        tree.create_node(val, str(add_to.identifier) + '_' + str(val), parent=add_to)
    return tree


def get_path_to_root_for(tree, leaf):
    path = []
    while not leaf.is_root():
        path.append(leaf.tag)
        leaf = tree.parent(leaf.identifier)
    path.append(leaf.tag)
    return path


def get_path_string(tree, leaf):
    path = ""
    while not leaf.is_root():
        path = '/' + str(leaf.tag) + path
        leaf = tree.parent(leaf.identifier)
    path = str(leaf.tag) + path
    return path


def get_minimal_subsets(sets):
    sets = sorted(map(set, sets), key=len)
    minimal_subsets = []
    minimal_subsets_list = []
    for s in sets:
        if not any(minimal_subset.issubset(s) for minimal_subset in minimal_subsets):
            minimal_subsets.append(s)
            minimal_subsets_list.append(list(s))
    return minimal_subsets_list
