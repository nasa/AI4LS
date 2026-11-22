import numpy as np


def join_keep_columns(list1,list2,alpha):
    # for symmetry in later stages
    if alpha>0.5:
        alpha = 1-alpha
        temp = list1
        list1 = list2
        list2 = list1
    
    n = len(list1)
    n1 = int(np.ceil(alpha*n))
    n2 = n-n1
    
    list_joint = list(set(list1[:n1] + list2[:n2]))
    
    count = 0
    for j in range(n-len(list_joint)):
        if list1[n1+count] not in list_joint:
            list_joint.append(list1[n1+count])
        count += 1
    
    return list_joint
    