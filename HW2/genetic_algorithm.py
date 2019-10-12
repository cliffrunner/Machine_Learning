import numpy as np
import random

def ga(max_iter=100, space=None, parent_size=20, fitness_func=None, scaling=10, problem='float'):

    assert fitness_func is not None
    assert space is not None

    idx = np.random.permutation(np.arange(len(space)))
    parents = space[idx[:parent_size]] # initialize parents

    for j in range(max_iter):
        print("genetic evoluation: {}|{}".format(j+1, max_iter))
        # print(parents)

        # select the best parents
        parents_score = [fitness_func(parent) for parent in parents]
        sort_index = np.argsort(parents_score)
        parent_size = len(parents)
        best_parents = parents[sort_index[int(parent_size/2):]] # argsort is ascending
        # parents = best_parents
        # print("best parents: {}".format(parents))

        # crossover
        if problem=='float':
            parents = produce_float_children(best_parents, space)
        elif problem=='binary':
            parents = produce_binary_children(best_parents, space)
        else:
            raise Exception("wrong problem category")

    parents_score = [fitness_func(parent) for parent in parents]
    sort_index = np.argsort(parents_score)
    return parents[sort_index[-1]]

def produce_float_children(best_parents, space):
    parents = best_parents
    if len(best_parents)%2==0:
        for i in range(int(len(best_parents)/2)):
            child1, child2 = mate_float(best_parents[i*2], best_parents[2*i+1])
            child1, child2 = mutate_float(child1), mutate_float(child2)
            if close_to_any(child1, space, 1e-12): parents = np.append(parents, child1)
            if close_to_any(child2, space, 1e-12): parents = np.append(parents, child2)
    else:
        for i in range(int(len(best_parents)/2)):
            child1, child2 = mate_float(best_parents[i*2], best_parents[2*i+1])
            child1, child2 = mutate_float(child1), mutate_float(child2)
            if close_to_any(child1, space, 1e-12): parents = np.append(parents, child1)
            if close_to_any(child2, space, 1e-12): parents = np.append(parents, child2)
        # let the last parent and the first parent mate
        child1, child2 = mate_float(best_parents[0], best_parents[-1])
        child1, child2 = mutate_float(child1), mutate_float(child2)
        if close_to_any(child1, space, 1e-12): parents = np.append(parents, child1)
        if close_to_any(child2, space, 1e-12): parents = np.append(parents, child2)
        # drop a random member
        parents = np.delete(parents, int(random.uniform(0, len(parents)-1)))
    return parents

def produce_binary_children(best_parents, space):
    parents = best_parents
    if len(best_parents)%2==0:
        for i in range(int(len(best_parents)/2)):
            child1, child2 = mate_binary(best_parents[i*2], best_parents[2*i+1])
            child1, child2 = mutate_binary(child1), mutate_binary(child2)
            if child1 in space: parents = np.append(parents, child1)
            if child2 in space: parents = np.append(parents, child2)
    else:
        for i in range(int(len(best_parents)/2)):
            child1, child2 = mate_binary(best_parents[i*2], best_parents[2*i+1])
            child1, child2 = mutate_binary(child1), mutate_binary(child2)
            if child1 in space: parents = np.append(parents, child1)
            if child2 in space: parents = np.append(parents, child2)
        # drop a random member
        parents = np.delete(parents, int(random.uniform(0, len(parents)-1)))
    return parents

def mate_float(parent1, parent2):
    decimal1 = parent1 - int(parent1)
    decimal2 = parent2 - int(parent2)
    child1 = int(parent1) + decimal2
    child2 = int(parent2) + decimal1
    return round(child1,1), round(child2,1)

def mutate_float(child):
    return round(child*(1+random.uniform(0,1)/10),1)

def mate_binary(parent1, parent2):
    child1 = parent1[:int(len(parent1)/2)] + parent2[int(len(parent2)/2):]
    child2 = parent2[:int(len(parent2)/2)] + parent1[int(len(parent1)/2):]
    assert len(child1) == len(child2)
    return child1, child2

def mutate_binary(child):
    # switch the bit 2nd to the last
    child = list(child)
    if child[int(len(child)/2)]=='1':
        child[int(len(child)/2)]='0'
    else:
        child[int(len(child)/2)]='1'
    return ''.join(child)

def close_to_any(a, floats, tol):
    return np.any(np.isclose(a, floats, tol))


def ga_chess(max_iter=200, parent_size=20, fitness_func=None, n=4):
    # generate parents pool
    parents = np.array([update_chess(n) for i in range(parent_size)])

    for j in range(max_iter):
        print("genetic evoluation: {}|{}".format(j+1, max_iter))
        # print(parents)

        # select the best parents
        parents_score = [fitness_func(parent) for parent in parents]
        sort_index = np.argsort(parents_score)
        print(parents_score)
        parent_size = len(parents)
        best_parents = parents[sort_index[int(parent_size/2):]] # argsort is ascending
        parents = produce_queen_children(best_parents)     

    parents_score = [fitness_func(parent) for parent in parents]
    sort_index = np.argsort(parents_score)
    parent = parents[sort_index[-1]]
    return parent

def update_chess(n):
    chess = np.zeros([n,n])
    ipos = []
    rows = []
    for i in range(10000):
        irow, icol = np.random.randint(0,n), np.random.randint(0,n)
        if irow not in rows: 
            ipos.append([irow, icol])
            rows.append(irow)
        if len(ipos) == n: break
    for pos in ipos:
        chess[pos[0],pos[1]] = 1
    return chess

def produce_queen_children(best_parents):
    parents = np.copy(best_parents)
    if len(best_parents)%2==0:
        for i in range(int(len(best_parents)/2)):
            child1, child2 = mate_queen(best_parents[i*2], best_parents[2*i+1])
            child1, child2 = mutate_queen(child1), mutate_queen(child2)
            parents = np.append(parents, np.array([child1]), axis=0)
            parents = np.append(parents, np.array([child2]), axis=0)
    else:
        for i in range(int(len(best_parents)/2)):
            child1, child2 = mate_queen(best_parents[i*2], best_parents[2*i+1])
            child1, child2 = mutate_queen(child1), mutate_queen(child2)
            parents = np.append(parents, np.array([child1]), axis=0)
            parents = np.append(parents, np.array([child2]), axis=0)
        # drop a random member
        parents = np.delete(parents, int(random.uniform(0, len(parents)-1)), axis=0)
    return parents

def mate_queen(parent1, parent2):
    child1 = np.concatenate((parent1[:int(len(parent1)/2)], parent2[int(len(parent2)/2):]), axis=0)
    child2 = np.concatenate((parent2[:int(len(parent2)/2)], parent1[int(len(parent1)/2):]), axis=0)
    return child1, child2

def mutate_queen(x):
    # only mutate along col
    n = len(x)
    pos_change = list(range(-int(n/2),int(n/2)+1))

    queens = np.where(x==1)
    # print(queens)
    queens = [[queens[0][i],queens[1][i]] for i in range(len(queens[0]))]

    cnt = 0
    while True:
        r = np.random.randint(0, len(queens))
        row, col= queens[r][0], queens[r][1]
        idx = np.random.choice(pos_change)
        if col+idx<=n-1 \
            and col+idx>=0 \
            and [row, col+idx] not in queens:
            queens[r] = [row, col+idx]
            cnt += 1
            if cnt==1: break
    chess = np.zeros([n,n])
    for q in queens:
        chess[q[0],q[1]] = 1

    return chess



if __name__=="__main__":
    """
    polynomial problem
    """
    # def fitness_func(x):
    #     return (x-11)*(x-5)*(x-14)*(x-3)*(x-13)*(x-9)*(x-6.4)*(x-18)*(x-18.5)*(x-2.8)*(x-19.1)

    # space = np.arange(2.2,18,0.1)

    # result = ga(max_iter=400, parent_size=20, fitness_func=fitness_func, space=space, scaling=10, problem='float')
    # print("best parents are: {}".format(result))


    """
    alternation problem
    """
    # def fitness_func(x):
    #     if '0b' in x:
    #         x = x[2:].zfill(bits)
    #     cnt = 0
    #     curr = x[0]
    #     for i, char in enumerate(x):
    #         if i==0: continue
    #         if char != curr:
    #             cnt += 1
    #             curr = char
    #     return cnt

    # step = 1
    # lower = 0
    # upper = 1023# 1048575 # binary for this is '1111111111', 10 bits
    # bits = 10

    # space = np.array([bin(x)[2:].zfill(bits) for x in np.arange(lower, upper, step)])
    # # print(space)

    # result = ga(max_iter=200, parent_size=60, fitness_func=fitness_func, space=space, scaling=10, problem='binary')

    # print("x for global maximum value is: {}".format(result))


    """
    n - queens problem
    """
    def fitness_func(x):
        n = len(x)
        score = 100
        # search queens
        queens = np.where(x==1)

        # check row conflicts
        score -= (len(queens[0]) - len(set(queens[0])))*2
        # check col conflicts
        score -= (len(queens[1]) - len(set(queens[1])))*2
        # check diagnal conflicts
        queens = [[queens[0][i],queens[1][i]] for i in range(len(queens[0]))]
        for pos in queens:
            search_se = [pos[0]+1, pos[1]+1]
            search_nw = [pos[0]-1, pos[1]-1]
            search_sw = [pos[0]+1, pos[1]-1]
            search_ne = [pos[0]-1, pos[1]+1]
            while True:
                # southeast
                if not np.all(np.isnan(search_se)):
                    if search_se[0] > n-1 or search_se[1] > n-1: 
                        search_se = np.nan
                    else:
                        if search_se in queens:
                            score -= 1
                        search_se = [search_se[0]+1, search_se[1]+1]
                # northwest
                if not np.all(np.isnan(search_nw)):
                    if search_nw[0] < 0 or search_nw[1] < 0:
                        search_nw = np.nan
                    else:
                        if search_nw in queens:
                            score -= 1
                        search_nw = [search_nw[0]-1, search_nw[1]-1]
                # southwest
                if not np.all(np.isnan(search_sw)):
                    if search_sw[0] > n-1 or search_sw[1] < 0:
                        search_sw = np.nan
                    else:
                        if search_sw in queens:
                            score -= 1
                        search_sw = [search_sw[0]+1, search_sw[1]-1]
                # northeast
                if not np.all(np.isnan(search_ne)):
                    if search_ne[0] < 0 or search_ne[1] > n-1:
                        search_ne = np.nan
                    else:
                        if search_ne in queens:
                            score -= 1
                        search_ne = [search_ne[0]-1, search_ne[1]+1]
                if np.all(np.isnan(search_se))\
                and np.all(np.isnan(search_nw))\
                and np.all(np.isnan(search_sw))\
                and np.all(np.isnan(search_ne)): break
        return score

    chess = ga_chess(max_iter=200, parent_size=10, fitness_func=fitness_func, n=8) 
    print(chess)
    print(fitness_func(chess))