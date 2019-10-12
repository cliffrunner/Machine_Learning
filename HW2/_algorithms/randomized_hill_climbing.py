import numpy as np
import matplotlib.pyplot as plt

def rhc(max_iter=50, fitness_func=None, space=None, step=1, opt=0, **args):
    assert fitness_func is not None
    assert space is not None

    if max_iter > len(space): max_iter = len(space)

    idx = np.random.permutation(np.arange(len(space)))

    globalx = space[idx[0]]
    for i in range(max_iter):
        curr = space[idx[i]]
        print("randomized search: {}|{}".format(i+1, max_iter))
        while True:
            maxx = curr
            left, right = find_neighbor(curr=curr, step=step)
            # print(left, right)
            neighbors = check_inbound(left, right, space)
            if isinstance(curr, str):
                curr = convert_binary(curr, bits=args['bits'])
                maxx = convert_binary(maxx, bits=args['bits'])
                neighbors = [convert_binary(n, bits=args['bits']) for n in neighbors]
            for neighbor in neighbors:
                if fitness_func(neighbor)>fitness_func(maxx):
                    maxx = neighbor  
            print(maxx, curr)   
            if is_equal(maxx, curr):
                if fitness_func(curr)>fitness_func(globalx):
                    globalx = curr
                break
            else:
                print("update {} to {}".format(curr, maxx))
                curr = maxx
                curr = rconvert_binary(curr)
        if isinstance(opt, str):
            pass
        else:
            if np.abs(globalx-opt) < 1e-6:
                break
    return i+1, globalx

def is_equal(i1, i2):
    if isinstance(i1, str):
        return i1==i2
    else:
        return np.abs(i1-i2) < 1e-6

def find_neighbor(curr=None, step=1):
    assert curr is not None
    
    if isinstance(curr, str):
        int_curr = int(curr,2)
        left = bin(int_curr-step)
        right = bin(int_curr+step)
    elif isinstance(curr, float):
        left = round(curr-step,1)
        right = round(curr+step,1)
    else:
        raise Exception("input type incorrect")
    
    return left, right

def check_inbound(left, right, space):
    results = []
    if left in space:
        results.append(left)
    if right in space:
        results.append(right)
    return results

def convert_binary(x, bits=10):
    return x[2:].zfill(bits)

def rconvert_binary(x):
    if isinstance(x, str): return bin(int(x,2))
    return x

def rhc_queen(max_iter=50, fitness_func=None, n=4):
    chess = np.zeros([n,n])
    # initial position generation
    ipos = []
    for i in range(10000):
        irow, icol = np.random.randint(0,n), np.random.randint(0,n)
        if [irow, icol] not in ipos: ipos.append([irow, icol])
        if len(ipos) == n: break
    for pos in ipos:
        chess[pos[0],pos[1]] = 1

    for i in range(max_iter):
        print("randomized search: {}|{}".format(i+1, max_iter))
        cur_score = fitness_func(chess)
        # find neighbor
        # randomly generate neighbor for a random queen
        new_chess = update_neighbor(chess)
        new_score = fitness_func(new_chess)
        print(new_score, cur_score)
        if new_score > cur_score:
            chess = new_chess
    return chess

def update_neighbor(x):
    n = len(x)
    pos_change = list(range(-int(n/2),int(n/2)+1))

    queens = np.where(x==1)
    # print(queens)
    queens = [[queens[0][i],queens[1][i]] for i in range(len(queens[0]))]

    cnt = 0
    while True:
        r = np.random.randint(0, len(queens))
        row, col = queens[r][0], queens[r][1]
        idx = [np.random.choice(pos_change), np.random.choice(pos_change)]
        if row+idx[0]<=n-1 \
            and row+idx[0]>=0 \
            and col+idx[1]<=n-1 \
            and col+idx[1]>=0 \
            and [row+idx[0], col+idx[1]] not in queens:
            queens[r] = [row+idx[0], col+idx[1]]
            cnt += 1
            if cnt==int(1): break
    # cnt = 0
    # while True:
    #     row, col = np.random.randint(0, len(queens)), np.random.randint(0, len(queens))
    #     if [row, col] not in queens: 
    #         queens.pop(-1)
    #         queens.append([row, col])
    #         cnt += 1
    #         if cnt==2: break
    chess = np.zeros([n,n])
    for q in queens:
        chess[q[0],q[1]] = 1

    return chess

if __name__=="__main__":

    """
    polynomial problem
    """
    def fitness_func(x):
        return (x-11)*(x-5)*(x-14)*(x-3)*(x-13)*(x-9)*(x-6.4)*(x-18)*(x-18.5)*(x-2.8)*(x-19.1)

    space = np.arange(2.2,18,0.1)

    iter_list = []
    results = []

    iteration, result = rhc(max_iter=50, fitness_func=fitness_func, space=space, step=0.1, opt=5.6)
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
    # upper = 1023 # binary for this is '11111111111111111111', 20 bits
    # bits=10

    # space = [bin(x) for x in np.arange(lower, upper, step)]

    # result = rhc(max_iter=10, fitness_func=fitness_func, space=space, step=1, bits=bits)
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

    # chess = rhc_queen(max_iter=5000, fitness_func=fitness_func, n=8)
    # print(chess)