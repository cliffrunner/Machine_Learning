import numpy as np
from sklearn.metrics import mutual_info_score

def mimic_binary(max_iter=100, fitness_func=None, space=None):

    assert fitness_func is not None
    assert space is not None

    idx = np.random.permutation(np.arange(len(space)))
    pool = space[idx[:int(len(space)/2)]] # randomly sample 50% of the oringal space

    new_pool = []

    for i in range(max_iter):
        print("mimic: {}|{}".format(i+1, max_iter))
        theta += delta
        for j, parent in enumerate(pool):
            if j in new_pool or fitness_func(parent)<theta: continue
            best_score = 0
            best_child = parent
            for k, child in enumerate(pool):
                if k<=j or child in new_pool: continue
                score = mutual_info(parent, child)
                if score > best_score and fitness_func(child)>=theta:
                    best_score = score
                    new_pool.append(parent)
                    new_pool.append(child)
    return None

def mutual_info(parent, child):
    parent = [int(x) for x in parent]
    child = [int(x) for x in child]
    return mutual_info_score(parent,child)