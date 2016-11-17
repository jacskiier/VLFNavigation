#!python
import numpy as np


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i + 1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0 + i) / n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j - 1)
    return indices


def particlefilter(sequence, pos, stepsize, n):
    seq = iter(sequence)
    x = np.ones((n, 2), int) * pos  # Initial position
    f0 = seq.next()[tuple(pos)] * np.ones(n)  # Target colour model
    yield pos, x, np.ones(n) / n  # Return expected position, particles and weights
    for im in seq:
        x += np.array(np.random.uniform(-stepsize, stepsize, x.shape), dtype=np.int32)  # Particle motion model: uniform step
        x = x.clip(np.zeros(2), np.array(im.shape) - 1).astype(int)  # Clip out-of-bounds particles
        f = im[tuple(x.T)]  # Measure particle colours
        w = 1. / (1. + (f0 - f) ** 2)  # Weight~ inverse quadratic colour distance
        w /= np.sum(w)  # Normalize w
        yield np.sum(x.T * w, axis=1), x, w  # Return expected position, particles and weights
        if 1. / np.sum(w ** 2) < n / 2.:  # If particle cloud degenerate:
            x = x[resample(w), :]  # Resample particles according to weights


if __name__ == "__main__":
    import pylab as plt
    from itertools import izip
    import time

    plt.ion()
    n_timesteps = 100
    n_particles = 1000
    noiseSigma = 10.0
    particleStepSize = 20.0
    seq = [im for im in np.zeros((n_timesteps, 240, 320), int)]  # Create an image sequence of 20 frames long

    x0 = np.array([120, 160])  # Add a square with starting position x0 moving along trajectory xs
    xs = np.vstack((np.arange(n_timesteps) * 3, np.arange(n_timesteps) * 2)).T + x0

    for timestep in range(1, n_timesteps):
        thisRandom = np.random.randn(2) * 10.0
        xs[timestep, :] = xs[timestep - 1, :] + thisRandom

    for t, x in enumerate(xs):
        xslice = slice(x[0] - 8, x[0] + 8)
        yslice = slice(x[1] - 8, x[1] + 8)
        seq[t][xslice, yslice] = 255

    for im, p in izip(seq, particlefilter(seq, x0, particleStepSize, n_particles)):  # Track the square through the sequence
        pos, xs, ws = p
        position_overlay = np.zeros_like(im)
        position_overlay[tuple(pos)] = 1
        particle_overlay = np.zeros_like(im)
        particle_overlay[tuple(xs.T)] = 1
        plt.hold(True)
        plt.draw()
        plt.cla()
        plt.imshow(im, cmap=plt.cm.get_cmap("gray"))  # Plot the image
        plt.spy(position_overlay, marker='.', color='b')  # Plot the expected position
        plt.spy(particle_overlay, marker=',', color='r')  # Plot the particles
        plt.show()
        plt.waitforbuttonpress(timeout=0.1)
