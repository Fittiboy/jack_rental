from math import factorial, exp
import matplotlib.pyplot as plt
from itertools import product
from datetime import datetime
import seaborn as sns
import numpy as np
import json


def get_value(g1, g2, a):
    gpa1, gpa2 = [g1-a, g2+a]
    aprobs = probs[gpa1, gpa2, :, :]
    r = rewards[gpa1, gpa2] - 2 * abs(a)
    return r + gamma * (V * aprobs).sum()


def eval(d_min):
    while True:
        d = 0
        for e1, e2 in product(range(ms), range(ms)):
            ov = V[e1, e2]
            vs = [get_value(e1, e2, ea) for ea in A[e1, e2]]
            V[e1, e2] = max(vs)
            d = max(d, abs(ov-V[e1, e2]))
        if d < d_min:
            break


def improve():
    for i1, i2 in product(range(ms), range(ms)):
        iqs = [(ia, get_value(i1, i2, ia)) for ia in A[i1, i2]]
        pi[i1, i2] = max(iqs, key=lambda x: x[1])[0]


def poisson(lmbd, n):
    return ((lmbd**n) / factorial(n)) * exp(-lmbd)


def post_return_prob(s_pret, s_preq):
    d_first = s_pret[0]-s_preq[0]
    if s_pret[0] == max_cars:
        if d_first == 0:
            p_first = 1
        else:
            p_first = 1 - sum(poissons["3"][i] for i in range(d_first))
    else:
        p_first = poissons["3"][d_first]
    d_second = s_pret[1]-s_preq[1]
    if s_pret[1] == max_cars:
        if d_second == 0:
            p_second = 1
        else:
            p_second = 1 - sum(poissons["2"][i] for i in range(d_second))
    else:
        p_second = poissons["2"][d_second]
    p = p_first * p_second
    return p


def post_request_prob(s_preq, s_pa):
    d_first = s_pa[0]-s_preq[0]
    if s_preq[0] == 0:
        if d_first == 0:
            p_first = 1
        else:
            p_first = 1 - sum(poissons["3"][i] for i in range(d_first))
    else:
        p_first = poissons["3"][d_first]
    d_second = s_pa[1]-s_preq[1]
    if s_preq[1] == 0:
        if d_second == 0:
            p_second = 1
        else:
            p_second = 1 - sum(poissons["4"][i] for i in range(d_second))
    else:
        p_second = poissons["4"][d_second]
    p = p_first * p_second
    return p


def get_reward(s_preq, s_pa):
    r_0 = 10 * (s_pa[0] - s_preq[0])
    r_1 = 10 * (s_pa[1] - s_preq[1])
    return r_0 + r_1


def p_4(s_pret, s_preq, s_pa):
    preq = post_request_prob(s_preq, s_pa)
    pret = post_return_prob(s_pret, s_preq)
    return preq*pret


if __name__ == "__main__":
    gamma = 0.9
    max_cars = 20
    ms = max_cars + 1
    mm = 5

    A = np.zeros((ms, ms), dtype=object)
    V = np.zeros((ms, ms), dtype=np.float64)
    pi = np.zeros((ms, ms), dtype=int)
    for n1, n2 in product(range(ms), range(ms)):
        A[n1, n2] = list(range(-min(mm, max_cars-n1, n2),
                               min(mm, max_cars-n2, n1)+1))

    lmbds = [2, 3, 4]
    poissons = {str(lmbd): [poisson(lmbd, n) for n in range(ms)]
                for lmbd in lmbds}

    print("Building tables")
    S_prets = np.zeros((21,), dtype=object)
    S_preqs = np.zeros((21,), dtype=object)
    for b in range(ms):
        S_pret = list(range(b, ms))
        S_preq = list(range(b+1))
        S_prets[b] = S_pret
        S_preqs[b] = S_preq

    probs = np.zeros((ms, ms, ms, ms))
    rewards = np.zeros((ms, ms))
    for n1, n2 in product(range(ms), range(ms)):
        print(f"State: [{n1}, {n2}]  ", end='\r')
        rewards[n1, n2] = 0
        for preq1, preq2 in product(S_preqs[n1], S_preqs[n2]):
            p_preq = post_request_prob([preq1, preq2], [n1, n2])
            rewards[n1, n2] += p_preq * get_reward([preq1, preq2],
                                                   [n1, n2])
            for pret1, pret2 in product(S_prets[preq1], S_prets[preq2]):
                p_pret = post_return_prob([pret1, pret2], [preq1, preq2])
                probs[n1, n2, pret1, pret2] += p_pret * p_preq
    print("Tables complete")

    start = datetime.now()
    for i in range(1000):
        print(f"VI Step {i+1: 4}", end="\r")
        V = np.zeros((ms, ms), dtype=np.float64)
        pi = np.zeros((ms, ms), dtype=int)
        eval(0.01)
        improve()
    runtime = datetime.now() - start
    print(f"Total runtime VI: {runtime}")
