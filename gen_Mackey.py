import numpy as np
import collections

def mackey_glass(sample_len=1000, tau=17, seed=None, n_samples = 1):

        delta_t = 10
        history_len = tau * delta_t 
        # Initial conditions for the history of the system
        timeseries = 1.2

        if seed is not None:
            np.random.seed(seed)

        samples = []

        for _ in range(n_samples):
            history = collections.deque(1.2 * np.ones(history_len) + 0.2 * \
                                        (np.random.rand(history_len) - 0.5))
            # Preallocate the array for the time-series
            inp = np.zeros((sample_len,1))
            
            for timestep in range(sample_len):
                for _ in range(delta_t):
                    xtau = history.popleft()
                    history.append(timeseries)
                    timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - \
                                    0.1 * history[-1]) / delta_t
                inp[timestep] = timeseries
            
            # Squash timeseries through tanh
            # inp = np.tanh(inp - 1)
            samples.append(inp)
        return samples