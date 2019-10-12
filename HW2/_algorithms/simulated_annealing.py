import numpy as np
import random

def sim_anneal(max_T=2000, min_T=50, T_step=1, fitness_func=None, space=None):
    assert fitness_func is not None
    assert space is not None

    max_iter = int((max_T-min_T)/T_step)
    curr_T = max_T

    idx = random.randint(0, len(space))
    curr = space[idx]

    for i in range(max_iter):
        print("simulated annuealing: {}|{}".format(i+1, max_iter))
        idx = random.randint(0, len(space)-1)
        if fitness_func(space[idx])>=fitness_func(curr):
            curr = space[idx]
        else:
            p = np.exp((fitness_func(space[idx])-fitness_func(curr))/curr_T)
            p_bench = random.uniform(0,1)
            if p>=p_bench:
                curr = space[idx]
        curr_T -= T_step
    return curr

def sim_anneal_queen(max_T=20, min_T=2, T_step=0.1, fitness_func=None, n=4):
    chess = update_chess(n)

    curr_T = max_T
    max_iter = int((max_T-min_T)/T_step)
    for i in range(max_iter):
        print("simulated annuealing: {}|{}".format(i+1, max_iter))
        new_chess = update_chess(n)
        print(fitness_func(new_chess),fitness_func(chess))
        if fitness_func(new_chess) > fitness_func(chess):
            chess = new_chess
        else:
            p = np.exp((fitness_func(new_chess)-fitness_func(chess))/curr_T)
            p_bench = random.uniform(0,1)
            if p>=p_bench:
                chess = new_chess
        curr_T -= T_step
    return chess

def update_chess(n):
    chess = np.zeros([n,n])
    ipos = []
    for i in range(10000):
        irow, icol = np.random.randint(0,n), np.random.randint(0,n)
        if [irow, icol] not in ipos: ipos.append([irow, icol])
        if len(ipos) == n: break
    for pos in ipos:
        chess[pos[0],pos[1]] = 1
    return chess

if __name__=="__main__":
    """
    polynomial problem
    """
    def fitness_func(x):
        return (x-11)*(x-5)*(x-14)*(x-3)*(x-13)*(x-9)*(x-6.4)*(x-18)*(x-18.5)*(x-2.8)*(x-19.1)

    space = np.arange(2.2,18,0.1)

    result = sim_anneal(max_T=1000000, min_T=5000, T_step=500, fitness_func=fitness_func, space=space)
    print("x for global maximum value is: {}".format(result))


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
    # upper = 1023 # binary for this is '1111111111', 10 bits
    # bits=10

    # space = [bin(x) for x in np.arange(lower, upper, step)]

    # result = sim_anneal(max_T=100, min_T=1, T_step=0.1, fitness_func=fitness_func, space=space)
    # print("x for global maximum value is: {}".format(result))


    """
    n - queens problem
    """
    # def fitness_func(x):
    #     n = len(x)
    #     score = 100
    #     # search queens
    #     queens = np.where(x==1)

    #     # check row conflicts
    #     score -= (len(queens[0]) - len(set(queens[0])))*2
    #     # check col conflicts
    #     score -= (len(queens[1]) - len(set(queens[1])))*2
    #     # check diagnal conflicts
    #     queens = [[queens[0][i],queens[1][i]] for i in range(len(queens[0]))]
    #     for pos in queens:
    #         search_se = [pos[0]+1, pos[1]+1]
    #         search_nw = [pos[0]-1, pos[1]-1]
    #         search_sw = [pos[0]+1, pos[1]-1]
    #         search_ne = [pos[0]-1, pos[1]+1]
    #         while True:
    #             # southeast
    #             if not np.all(np.isnan(search_se)):
    #                 if search_se[0] > n-1 or search_se[1] > n-1: 
    #                     search_se = np.nan
    #                 else:
    #                     if search_se in queens:
    #                         score -= 1
    #                     search_se = [search_se[0]+1, search_se[1]+1]
    #             # northwest
    #             if not np.all(np.isnan(search_nw)):
    #                 if search_nw[0] < 0 or search_nw[1] < 0:
    #                     search_nw = np.nan
    #                 else:
    #                     if search_nw in queens:
    #                         score -= 1
    #                     search_nw = [search_nw[0]-1, search_nw[1]-1]
    #             # southwest
    #             if not np.all(np.isnan(search_sw)):
    #                 if search_sw[0] > n-1 or search_sw[1] < 0:
    #                     search_sw = np.nan
    #                 else:
    #                     if search_sw in queens:
    #                         score -= 1
    #                     search_sw = [search_sw[0]+1, search_sw[1]-1]
    #             # northeast
    #             if not np.all(np.isnan(search_ne)):
    #                 if search_ne[0] < 0 or search_ne[1] > n-1:
    #                     search_ne = np.nan
    #                 else:
    #                     if search_ne in queens:
    #                         score -= 1
    #                     search_ne = [search_ne[0]-1, search_ne[1]+1]
    #             if np.all(np.isnan(search_se))\
    #             and np.all(np.isnan(search_nw))\
    #             and np.all(np.isnan(search_sw))\
    #             and np.all(np.isnan(search_ne)): break
    #     return score

    # chess = sim_anneal_queen(max_T=2, min_T=0.001, T_step=0.001, fitness_func=fitness_func, n=4)
    # print(chess)

