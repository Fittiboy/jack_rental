from math import factorial, exp
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


def eval(S, A, p, gamma, V, pi, d_min):
    while True:
        d = 0
        for s in S:
            print(f"Evaluating {s}  ", end='\r')
            v = V[s]
            V[s] = get_value(s, pi[s], S, p, gamma, V)
            d = max(d, abs(v-V[s]))
        if d < d_min:
            break
    return V


def improve(S, A, p, gamma, V, pi):
    stable = True
    n_changed = 0
    for s in S:
        changed = False
        old = V[s]
        print(f"Finding best action for state {s}  ", end='\r')
        for a in A[s]:
            new = get_value(s, a, S, p, gamma, V)
            if new > old:
                if pi[s] != a:
                    changed = True
                    stable = False
                    pi[s] = a
        if changed:
            n_changed += 1
    print(f"\nActions changed: {n_changed}")
    return stable, pi


def poisson(lmbd, n):
    return ((lmbd**n) / factorial(n)) * exp(-lmbd)


def post_return_prob(s_pret, s_pa):
    p = poissons["3"][s_pret[0]-s_pa[0]] * poissons["2"][s_pret[1]-s_pa[1]]
    return p


def post_request_prob(s_preq, s_pret):
    p = poissons["3"][s_pret[0]-s_preq[0]] * poissons["4"][s_pret[1]-s_preq[1]]
    return p


def get_reward(s_preq, s_pret, a):
    r_0 = 10 * (s_pret[0] - s_preq[0])
    r_1 = 10 * (s_pret[1] - s_preq[1])
    return r_0 + r_1 - (2 * a)


def p_4(s_next, s_pret, s_pa):
    probs = []
    preq = post_request_prob(s_next, s_pret)
    pret = post_return_prob(s_pret, s_pa)
    probs.append(preq*pret)
    return sum(probs)


if __name__ == "__main__":
    max_cars = 20
    ms = max_cars + 1
    mm = 5

    S = {f"{n1}, {n2}": [n1, n2] for n1 in range(ms) for n2 in range(ms)}
    A = {f"{n1}, {n2}": [a for a in range(min(mm, 20-n2, n1)+1)] +
         [-a for a in range(1, min(mm, 20-n1, n2)+1)] for n1 in range(ms) for
         n2 in range(ms)}
    gamma = 0.9

    V, pi = initialize(S, A)

    lmbds = [2, 3, 4]
    ns = [i for i in range(ms)]
    poissons = {str(lmbd): [poisson(lmbd, n) for n in ns] for lmbd in lmbds}

    try:
        with open("prets.json") as prets_file:
            S_prets = json.load(prets_file)
        with open("probs.json") as probs_file:
            probs = json.load(probs_file)
    except FileNotFoundError:
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
        with open("prets.json", "w") as prets_file:
            json.dump(S_prets, prets_file, indent=4)
        with open("probs.json", "w") as probs_file:
            json.dump(probs, probs_file, indent=4)
        print("Tables complete")

    stable = False
    i = 0
    while not stable:
        i += 1
        print(f"Step: {i}")
        V = eval(S, A, probs, gamma, V, pi, 1)
        print("Policy evaluated.")
        stable, pi = improve(S, A, probs, gamma, V, pi)
        print("Policy updated.")

    print(pi)
