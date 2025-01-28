import numpy as np

rng = np.random.RandomState(42)


# Define the CV dynamics model
def cv_dynamics(x, A):
    new_state = A @ x
    return new_state.astype(int)


def gen_states(x0, A, num_states):
    x = [x0]
    for idx in range(1, num_states, 1):
        xk = cv_dynamics(x[idx - 1], A)
        x.append(xk)

    return np.array(x)
