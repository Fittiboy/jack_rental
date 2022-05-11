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


if __name__ == "__main__":
    S = ["1", "2", "3"]
    A = {"1": ["3", "2", "1"],
         "2": ["3", "1", "2"],
         "3": ["2", "1", "3"]}
    R = [1, 2, 3]
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
