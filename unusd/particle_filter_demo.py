import numpy as np
import matplotlib.pyplot as plt

# Simulate a moving square
def moving_square(x, t):
    return x + t

# Particle Filter implementation
def particle_filter(y, n_particles, n_iterations):
    particles = np.random.uniform(0, 100, n_particles)
    weights = np.ones(n_particles) / n_particles
    estimates = []

    for t in range(n_iterations):
        # Propagate particles
        particles = moving_square(particles, 1) + np.random.normal(0, 1, n_particles)

        # Update weights based on the likelihood of each particle
        weights = np.exp(-0.5 * ((y[t] - particles) ** 2))  # Gaussian likelihood
        weights /= np.sum(weights)

        # Resample particles
        indices = np.random.choice(np.arange(n_particles), size=n_particles, p=weights)
        particles = particles[indices]

        # Compute the estimate as the mean of the particles
        estimates.append(np.mean(particles))

    return estimates

# Generate the true trajectory and observations with noise
n_iterations = 50
true_trajectory = [moving_square(10, t) for t in range(n_iterations)]
observations = [y + np.random.normal(0, 1) for y in true_trajectory]

# Apply Particle Filter
n_particles = 1000
estimates = particle_filter(observations, n_particles, n_iterations)

# Plot the results
plt.figure()
plt.plot(true_trajectory, label='True trajectory')
plt.plot(observations, 'o', label='Noisy observations')
plt.plot(estimates, label='Particle Filter estimates')
plt.legend()
plt.show()