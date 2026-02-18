import numpy as np


class ParticleFilter:
    def __init__(self, N, sensor):
        self.N = N
        self.sensor = sensor
        self.particles, self.weights = self.initialize_particles()

    # 1. Initialization (at k = 0)
    def initialize_particles(self):
        particles = np.zeros((self.N, 3))

        # Sample from prior (uniform over map)
        particles[:, 0] = np.random.uniform(0, 10, self.N)      # x
        particles[:, 1] = np.random.uniform(0, 8, self.N)       # y
        particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.N)  # theta

        weights = np.ones(self.N) / self.N  # equal weight, unfiform proabilty of being anywhere

        return particles, weights

    #2. Prediction (motion update)
    def motion_update(self, control, noise_std):
        v, omega = control

        N = len(self.particles)

        # Add noise to control
        v_noisy = v + np.random.normal(0, noise_std[0], N)
        omega_noisy = omega + np.random.normal(0, noise_std[1], N)

        # Update state apply map to output to (x,y,theta)
        self.particles[:, 0] += v_noisy * np.cos(self.particles[:, 2])
        self.particles[:, 1] += v_noisy * np.sin(self.particles[:, 2])
        self.particles[:, 2] += omega_noisy

    # 3. Measurement update (weighting)
    def measurement_update(self, measurements, sigma):
        for i, p in enumerate(self.particles):
            for angle, z in zip(self.sensor.sensor_angles, measurements):
                predicted = self.sensor.measure(p, angle_offset=angle)
                error = z - predicted
                #likelihood = np.exp(-(error ** 2) / (2 * sigma ** 2))
                likelihood = self.sensor.pz(z, predicted, sigma=sigma)
                self.weights[i] *= likelihood

        # Avoid division by zero
        self.weights += 1e-300

        # Normalize
        self.weights /= np.sum(self.weights)

    # 5. Resampling
    def resample(self, replace_ratio=0.0005):
        N = len(self.particles)

        # If weights are too uniform (no good matches), inject random particles
        n_eff = 1.0 / np.sum(self.weights**2)  # Effective sample size
        n_random = int(N * replace_ratio)

        # Resample based on weights
        indices = np.random.choice(N, size=N, p=self.weights)
        self.particles = self.particles[indices]

        # Replace some particles with random ones to avoid particle deprivation
        if n_eff < N * 0.1:  # If effective particles < 10% of total
            n_random = int(N * 0.2)  # Replace 20% instead

        # Get map bounds from walls
        walls = self.sensor.walls
        x_min = min(w[0] for w in walls)
        x_max = max(w[2] for w in walls)
        y_min = min(w[1] for w in walls)
        y_max = max(w[3] for w in walls)

        # Replace last n_random particles with random poses
        random_indices = np.random.choice(N, size=n_random, replace=False)
        for i in random_indices:
            self.particles[i, 0] = np.random.uniform(x_min, x_max)
            self.particles[i, 1] = np.random.uniform(y_min, y_max)
            self.particles[i, 2] = np.random.uniform(-np.pi, np.pi)

        self.weights = np.ones(N) / N

    # 6. State estimation (from the particles right now, where is the robot)
    # just for us to see where thing robots
    def estimate_state(self):
        #expection value for x,y,theta
        x = np.average(self.particles[:, 0], weights=self.weights)
        y = np.average(self.particles[:, 1], weights=self.weights)

        cos_sum = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        sin_sum = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        theta = np.arctan2(sin_sum, cos_sum)

        return np.array([x, y, theta])

    # 7. Localization detection
    def is_localized(self, pos_threshold=0.5, angle_threshold=0.3):
        x_var = np.var(self.particles[:, 0])
        y_var = np.var(self.particles[:, 1])

        # Circular variance for angle
        cos_mean = np.mean(np.cos(self.particles[:, 2]))
        sin_mean = np.mean(np.sin(self.particles[:, 2]))
        angle_var = 1 - np.sqrt(cos_mean**2 + sin_mean**2)

        return (x_var < pos_threshold and
                y_var < pos_threshold and
                angle_var < angle_threshold)