import numpy as np

def uncertainty(label_probs,n_points,measure):
    if measure=='least confident':
        return np.argsort([1-max(y) for y in label_probs])[-n_points:]
    if measure=='margin':
        return np.argsort([np.sort(y)[-1]-np.sort(y)[-2] for y in label_probs])[:n_points]
    if measure=='entropy':
        return np.argsort([-sum(y*np.log(y)) for y in label_probs])[-n_points:]