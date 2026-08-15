import numpy as np
import matplotlib.pyplot as plt

H = 10
pi_H = np.zeros((H, 7), dtype = int)
V_H = np.zeros((H + 1, 7))
v0 = np.zeros(7)
P = np.zeros((2, 7, 7))
P[0] = np.eye(7)
P[1] = np.array([
    [0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.4, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.4, 0.2, 0.4, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.4, 0.2, 0.4, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.4, 0.2, 0.4, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.4, 0.2, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.6]
])
reward = np.zeros((7, 2))
reward[0, :] = 1
reward[6, :] = 10
gamma = 0.54

def vh_find():
    v_prev = v0
    for k in range(1, H + 1):
        Q_s = reward + gamma * (np.dot(P, v_prev).T)
        v_fn = np.max(Q_s, axis = 1)
        V_H[k] = v_fn
        pi_H[k-1] = np.argmax(Q_s, axis = 1)
        v_prev = v_fn

def sim_ep(start_state):
    curr = start_state
    G = 0.0
    for t in range(H):
        action = pi_H[H-t-1][curr]
        G += (gamma ** t) * (reward[curr, action])
        curr = np.random.choice(np.arange(7), p = P[action, curr, :])
    return G

def loop_prf():
    ret = []
    for _ in range(10000):
        G_t = sim_ep(start_state = 3)
        ret.append(G_t)
    return np.mean(ret), ret


if __name__ == "__main__":
    vh_find()
    start = np.random.choice(7)
    sim_ep(start)
    G_t, ret = loop_prf()
    print(V_H[H][3])
    print(G_t)
    

    