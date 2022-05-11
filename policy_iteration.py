from math import factorial, exp


def initialize(S: list, A: dict):
    V = {s: 0 for s in S}
    pi = {s: A[s[0]] for s in S}
    return V, pi


def get_value(s, a, S, R, p, gamma, V):
    return sum(p(s_next, r, s, a)*(r+gamma*V[s_next])
               for s_next in S for r in R)


def eval(S, A, R, p, gamma, V, pi, d_min):
    while True:
        d = 0
        for s in S:
            v = V[s]
            V[s] = get_value(s, pi[s], S, R, p, gamma, V)
            d = max(d, abs(v-V[s]))
        if d < d_min:
            break
    return V


def improve(S, A, R, p, gamma, V, pi):
    stable = True
    for s in S:
        old = get_value(s, pi[s], S, R, p, gamma, V)
        for a in A:
            if get_value(s, a, S, R, p, gamma, V) > old:
                pi[s] = a
                stable = False
    return stable, pi


def poisson(lmbd, n):
    return ((lmbd**n) / factorial(n)) * exp(-lmbd)


def post_return_prob(s_pret, s, a):
    s_pa = [s[0]-a, s[0]+a]
    if s_pret[0] < s_pa[0] or s_pret[1] < s_pa[0]:
        return 0
    else:
        p = poisson(3, s_pret[0]-s_pa[0]) * poisson(2, s_pret[1]-s_pa[1])
        return p


def post_request_prob(s_preq, s_pret):
    if s_preq[0] > s_pret[0] or s_preq[1] > s_pret[1]:
        return 0
    else:
        p = poisson(3, s_pret[0]-s_preq[0]) * poisson(4, s_pret[1]-s_preq[1])
        return p


def reward_prob(r, s_preq, s_pret, a):
    r_0 = 10 * (s_pret[0] - s_preq[0])
    r_1 = 10 * (s_pret[1] - s_pret[1])
    r_true = r_0 + r_1 - (2 * a)
    return 1 if r == r_true else 0


def p_4(S, s_next, r, s, a):
    return sum(post_request_prob(s_next, s_pret)*post_return_prob(s_pret, s, a)
               for s_pret in S)


if __name__ == "__main__":
    S = {f"{n1}, {n2}": [n1, n2] for n1 in range(21) for n2 in range(21)}
    A = {f"{n1}, {n2}": [a for a in range(1, min(5, 20-n2, n1)+1)] for n1 in
         range(21) for n2 in range(21)}
    R = list(range(-10, 401, 2))
    gamma = 0.9

    def p(s_next, r, s, a):
        if s_next == a:
            r_true = 1 if s_next == "3" else 0
            return r if r_true == r else 0
        else:
            return 0

    V, pi = initialize(S, A)

    stable = False
    while not stable:
        V = eval(S, A, R, p, gamma, V, pi, 0.000001)
        stable, pi = improve(S, A, R, p, gamma, V, pi)

    print(pi)
