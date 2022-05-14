from math import factorial, exp, inf
from datetime import datetime
import json


def initialize(S: dict, A: dict):
    V = {s: 0 for s in S}
    pi = {s: A[s][0] for s in S}
    return V, pi


def get_value(s, a, S, p, gamma, V):
    rets = []
    s_pa = [S[s][0]-a, S[s][1]+a]
    for s_preq in S:
        for s_pret in S_prets[str(s_pa)][s_preq]:
            r = get_reward(S[s_preq], s_pret, a)
            prob = probs[str(s_pa)][s_preq][str(s_pret)]
            ret = r+gamma*V[s_preq]
            rets.append(prob*ret)
    return sum(rets)


def eval(S, A, p, gamma, V, d_min):
    while True:
        d = 0
        for s in S:
            print(f"Evaluating {s}  ", end='\r')
            v = V[s]
            old_v = v
            for a in A[s]:
                new_v = get_value(s, a, S, p, gamma, V)
                if new_v > old_v:
                    old_v = new_v
            V[s] = new_v
            d = max(d, abs(v-V[s]))
        if d < d_min:
            break
    return V


def improve(S, A, p, gamma, V, pi):
    for s in S:
        print(f"Finding best action for state {s}  ", end='\r')
        best = -inf
        for a in A[s]:
            q = get_value(s, a, S, p, gamma, V)
            if q > best:
                best = q
                pi[s] = a
    print()
    return pi


def poisson(lmbd, n):
    return ((lmbd**n) / factorial(n)) * exp(-lmbd)


def post_return_prob(s_pret, s_pa):
    d_first = s_pret[0]-s_pa[0]
    if s_pret[0] == max_cars:
        if d_first == 0:
            p_first = 1
        else:
            p_first = 1 - sum(poissons["3"][i] for i in range(d_first))
    else:
        p_first = poissons["3"][d_first]
    d_second = s_pret[1]-s_pa[1]
    if s_pret[1] == max_cars:
        if d_second == 0:
            p_second = 1
        else:
            p_second = 1 - sum(poissons["2"][i] for i in range(d_second))
    else:
        p_second = poissons["2"][d_second]
    p = p_first * p_second
    return p


def post_request_prob(s_preq, s_pret):
    d_first = s_pret[0]-s_preq[0]
    if s_preq[0] == 0:
        if d_first == 0:
            p_first = 1
        else:
            p_first = 1 - sum(poissons["3"][i] for i in range(d_first))
    else:
        p_first = poissons["3"][d_first]
    d_second = s_pret[1]-s_preq[1]
    if s_preq[1] == 0:
        if d_second == 0:
            p_second = 1
        else:
            p_second = 1 - sum(poissons["4"][i] for i in range(d_second))
    else:
        p_second = poissons["4"][d_second]
    p = p_first * p_second
    return p


def get_reward(s_preq, s_pa, a):
    r_0 = 10 * (s_pa[0] - s_preq[0])
    r_1 = 10 * (s_pa[1] - s_preq[1])
    return r_0 + r_1 - (2 * abs(a))


def p_4(s_next, s_pret, s_pa):
    preq = post_request_prob(s_next, s_pret)
    pret = post_return_prob(s_pret, s_pa)
    return preq*pret


if __name__ == "__main__":
    max_cars = 8
    ms = max_cars + 1
    mm = 2

    S = {f"{n1}, {n2}": [n1, n2] for n1 in range(ms) for n2 in range(ms)}
    A = {f"{n1}, {n2}": [a for a in range(min(mm, max_cars-n2, n1)+1)] +
         [-a for a in range(1, min(mm, max_cars-n1, n2)+1)] for n1 in range(ms)
         for n2 in range(ms)}
    gamma = 0.9

    V, pi = initialize(S, A)

    lmbds = [2, 3, 4]
    ns = [i for i in range(ms)]
    poissons = {str(lmbd): [poisson(lmbd, n) for n in ns] for lmbd in lmbds}

    try:
        print("Trying to load tables...")
        with open(f"prets_{ms}_{mm}.json") as prets_file:
            S_prets = json.load(prets_file)
        with open(f"probs_{ms}_{mm}.json") as probs_file:
            probs = json.load(probs_file)
        print("Tables loaded!")
    except FileNotFoundError:
        print("No precomputed tables.")
        print("Building tables")
        S_prets = {}
        probs = {}
        for s in S:
            print(f"State: {s}  ", end='\r')
            for a in A[s]:
                s_pa = [S[s][0]-a, S[s][1]+a]
                S_prets[str(s_pa)] = {}
                probs[str(s_pa)] = {}
                for s_preq in S:
                    lb0 = max(s_pa[0], S[s_preq][0])
                    lb1 = max(s_pa[1], S[s_preq][1])
                    S_pret = [[n1, n2] for n1 in range(lb0, ms) for n2 in
                              range(lb1, ms)]
                    probs[str(s_pa)][s_preq] = {}
                    for s_pret in S_pret:
                        probs[str(s_pa)][s_preq][str(s_pret)] = p_4(S[s_preq],
                                                                    s_pret,
                                                                    s_pa)
                    S_prets[str(s_pa)][s_preq] = S_pret
        with open(f"prets_{ms}_{mm}.json", "w") as prets_file:
            json.dump(S_prets, prets_file, indent=4)
        with open(f"probs_{ms}_{mm}.json", "w") as probs_file:
            json.dump(probs, probs_file, indent=4)
        print("Tables complete")

    start = datetime.now()
    V = eval(S, A, probs, gamma, V, 0.0001)
    print("Value iteration complete!")
    pi = improve(S, A, probs, gamma, V, pi)
    print("Policy updated.")
    runtime = datetime.now() - start
    print(f"Total runtime: {runtime}")

    with open(f"vi_policy_{ms}_{mm}.json", "w") as policy_file:
        json.dump(pi, policy_file, indent=4)
    print("Policy saved.")
    with open(f"vi_value_{ms}_{mm}.json", "w") as value_file:
        json.dump(V, value_file, indent=4)
    print("Values saved.")
