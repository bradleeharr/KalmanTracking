import numpy as np

class ParticleFilter:

    def __init__(self, n_particles, state_dim, measurement_dim):
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.particles = np.random.uniform(size=(n_particles, state_dim))
        self.weights = np.ones(n_particles) / n_particles

    def predict(self, motion_model, noise_covariance):
        for i in range(self.n_particles):
            self.particles[i] = motion_model(self.particles[i])
            self.particles[i] += np.random.multivariate_normal(np.zeros(self.state_dim), noise_covariance)

    def update(self, measurement, likelihood_function):
        for i in range(self.n_particles):
            self.weights[i] = likelihood_function(measurement, self.particles[i])

        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(np.arange(self.n_particles), size=self.n_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles

    def estimate(self):
        return np.mean(self.particles, axis=0)