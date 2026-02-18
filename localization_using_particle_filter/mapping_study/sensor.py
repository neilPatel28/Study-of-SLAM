import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class Sensor:
    def __init__(self, walls, sensor_angles=None):
        self.walls = walls
        self.sensor_angles = sensor_angles or [0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2]

    # simulate a sensor
    def sensor(self, ray_origin, ray_dir, p1, p2):
        x, y = ray_origin
        dx, dy = ray_dir

        x1, y1 = p1
        x2, y2 = p2

        sx = x2 - x1
        sy = y2 - y1

        denom = dx * sy - dy * sx

        if abs(denom) < 1e-8:
            return None  # parallel

        t = ((x1 - x) * sy - (y1 - y) * sx) / denom
        u = ((x1 - x) * dy - (y1 - y) * dx) / denom

        if t > 0 and 0 <= u <= 1:
            return t  # distance along ray

        return None

    def pz(self, z, z_true, sigma=0.5, lam=1.0, z_max=20.0, eps=0.1,
           w_hit=0.70, w_short=0.10, w_max=0.10, w_rand=0.10):
        """p(z | x_t, m) — probability density of measurement z given true range."""

        # p_hit: gaussian centered on true range, truncated to [0, z_max]
        if 0 <= z <= z_max:
            g = np.exp(-0.5 * ((z - z_true) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            eta_hit = norm.cdf(z_max, z_true, sigma) - norm.cdf(0, z_true, sigma)
            hit = g / eta_hit if eta_hit > 1e-12 else 0.0
        else:
            hit = 0.0

        # p_short: exponential truncated to [0, z_true]
        if 0 <= z <= z_true and z_true > 1e-6:
            eta_short = 1.0 - np.exp(-lam * z_true)
            short = (lam * np.exp(-lam * z)) / eta_short if eta_short > 1e-12 else 0.0
        else:
            short = 0.0

        # p_max: narrow peak at max range
        mx = 1.0 / (2 * eps) if abs(z - z_max) <= eps else 0.0

        # p_rand: uniform over [0, z_max]
        rand = 1.0 / z_max if 0 <= z <= z_max else 0.0

        return w_hit * hit + w_short * short + w_max * mx + w_rand * rand

    def measure(self, pose, angle_offset=0, max_range=20):
        x, y, theta = pose
        ray_origin = np.array([x, y])
        ray_dir = np.array([np.cos(theta + angle_offset), np.sin(theta + angle_offset)])
        min_dist = max_range

        for x1, y1, x2, y2 in self.walls:
            dist = self.sensor(ray_origin, ray_dir, np.array([x1, y1]), np.array([x2, y2]))

            if dist is not None and dist < min_dist:
                min_dist = dist

        return min_dist

    # we made pz based on the noise of real world, this just adds noise
    def noisy_measure(self, pose, angle_offset=0, max_range=20,
                      sigma=0.05, lam=1.0,
                      w_hit=0.70, w_short=0.10, w_max=0.10, w_rand=0.10):
        z_true = self.measure(pose, angle_offset, max_range)

        # Pick which noise source
        r = np.random.random()

        if r < w_hit:
            # Gaussian around true range
            z = np.random.normal(z_true, sigma)
            z = np.clip(z, 0, max_range)

        elif r < w_hit + w_short:
            # Unexpected short reading (exponential)
            z = np.random.exponential(1.0 / lam)
            if z > z_true:
                z = z_true  # truncated to [0, z_true]

        elif r < w_hit + w_short + w_max:
            # Max range failure
            z = max_range

        else:
            # Random nonsense
            z = np.random.uniform(0, max_range)

        return z

    def draw_sensors(self, pose, max_range=20):
        x, y, theta = pose
        for angle in self.sensor_angles:
            dist = self.measure(pose, angle_offset=angle, max_range=max_range)
            end_x = x + dist * np.cos(theta + angle)
            end_y = y + dist * np.sin(theta + angle)
            plt.plot([x, end_x], [y, end_y], 'g-', alpha=0.5)
            plt.plot(end_x, end_y, 'go', markersize=3)