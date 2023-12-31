import numpy as np
import matplotlib.pyplot as plt

def spherical3d(R, THETA, PHI):
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    fig = plt.figure(figsize=plt.figaspect(1)*4)  # Square figure
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z)
    return fig

def cylindrical3d(S, PHI, Z):
    X = S * np.cos(PHI)
    Y = S * np.sin(PHI)

    fig = plt.figure(figsize=plt.figaspect(1)*4)  # Square figure
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z)
    return fig
