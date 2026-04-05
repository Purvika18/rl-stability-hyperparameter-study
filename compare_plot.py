import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, window=20):
    x = np.array(x, dtype=np.float32)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="valid")

print("Next: we will modify reinforce_agent.py to save rewards to a .npy file, then plot here.")