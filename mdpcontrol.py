import numpy as np


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

reward = np.zeros((7,2))
reward[0, :] = 1.0
reward[6, :] = 10.0

v0 = np.zeros(7)
gamma = 0.7


def policy_eval(pi):
    P_pi = P[pi, np.arange(7), :]
    R_pi = reward[np.arange(7), pi]
    v_prev = np.zeros(7) 
    val_fn = np.zeros(7)
    
    for _ in range(1, 1000):
        val_fn = R_pi + gamma * np.dot(P_pi, v_prev)
        if np.max(np.abs(val_fn - v_prev)) <= 0.0001:
            v_prev = val_fn
            break
        v_prev = val_fn
    return np.round(val_fn, 2)

def policy_improv(V_pi):
    Q_sa = reward + gamma * (np.dot(P, V_pi).T)
    Q_sa = np.argmax(Q_sa, axis=1)
    return Q_sa

def val_iter():
    v_prev = np.zeros(7)
    for _ in range(1, 1000):
        Q = reward + gamma * (np.dot(P, v_prev).T)
        V_n = np.max(Q, axis=1)
        if np.max(np.abs(V_n - v_prev)) <= 0.000001:
            v_prev = V_n
            break
        v_prev = V_n
        
    Q_fin = reward + gamma * (np.dot(P, v_prev).T)
    op_pi = np.argmax(Q_fin, axis=1)
    
    return np.round(v_prev, 2), op_pi

if __name__ == "__main__":
    pi = np.random.choice([0,1], size=7)
    curr_pi = np.copy(pi)
    
    iterations = 0 
    
    print(f"Initial Random Policy: {pi}\n")
    
    while True:
        iterations += 1
        V_pi = policy_eval(curr_pi)
        new_pi = policy_improv(V_pi)

        if np.array_equal(new_pi, curr_pi):
            print(f"Policy Iteration Converged in {iterations} step(s).")
            print(f"Optimal Policy (PI): {new_pi}")
            print(f"Optimal Values (PI): {V_pi}")
            break
        curr_pi = new_pi

    print("\n--------------------------\n")

    V_star, pi_star = val_iter()
    print("Value Iteration Completed.")
    print(f"Optimal Policy (VI): {pi_star}")
    print(f"Optimal Values (VI): {V_star}")