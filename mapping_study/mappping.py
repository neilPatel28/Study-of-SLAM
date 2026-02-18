import numpy as np
import matplotlib.pyplot as plt


class OccupancyGrid:
    def __init__(self, x_min, x_max, y_min, y_max, resolution=0.1):
        self.resolution = resolution
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        # Grid dimensions
        self.width = int(np.ceil((x_max - x_min) / resolution))
        self.height = int(np.ceil((y_max - y_min) / resolution))

        # Log-odds grid (0 = unknown prior)
        self.log_odds = np.zeros((self.height, self.width))

        # Prior probability p=0.5 => log-odds = 0
        self.l0 = 0.0

        # Inverse sensor model parameters
        """self.l_occ = np.log(0.7 / 0.3)   # occupied update
        self.l_free = np.log(0.3 / 0.7)   # free update"""
        self.l_occ = np.log(0.9 / 0.1)
        self.l_free = np.log(0.1 / 0.9)

        # Clamp log-odds to avoid saturation
        self.l_min = -5.0
        self.l_max = 5.0

    def world_to_grid(self, x, y):
        gx = int((x - self.x_min) / self.resolution)
        gy = int((y - self.y_min) / self.resolution)
        gx = np.clip(gx, 0, self.width - 1)
        gy = np.clip(gy, 0, self.height - 1)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = gx * self.resolution + self.x_min + self.resolution / 2
        y = gy * self.resolution + self.y_min + self.resolution / 2
        return x, y

    def bresenham(self, x0, y0, x1, y1):
        """Bresenham's line algorithm — returns list of (gx, gy) cells along ray."""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return cells

    def inverse_sensor_model(self, pose, z, angle_offset, max_range=20.0):

        x, y, theta = pose
        ray_angle = theta + angle_offset

        # Robot cell
        rx, ry = self.world_to_grid(x, y)

        # Endpoint of measurement
        hit = z < max_range - 0.1
        end_x = x + z * np.cos(ray_angle)
        end_y = y + z * np.sin(ray_angle)
        ex, ey = self.world_to_grid(end_x, end_y)

        # Bresenham ray trace from robot to endpoint
        cells = self.bresenham(rx, ry, ex, ey)

        # All cells except the last are free
        for gx, gy in cells[:-1]:
            if 0 <= gx < self.width and 0 <= gy < self.height:
                # Incremental log-odds update: l_t = l_{t-1} + l_free - l_0
                self.log_odds[gy, gx] += self.l_free - self.l0
                self.log_odds[gy, gx] = np.clip(self.log_odds[gy, gx], self.l_min, self.l_max)

        # Last cell is occupied (if we actually hit something)
        if hit and 0 <= ex < self.width and 0 <= ey < self.height:
            # Incremental log-odds update: l_t = l_{t-1} + l_occ - l_0
            self.log_odds[ey, ex] += self.l_occ - self.l0
            self.log_odds[ey, ex] = np.clip(self.log_odds[ey, ex], self.l_min, self.l_max)


    #update grid
    def update(self, pose, measurements, sensor_angles, max_range=20.0):
        for z, angle in zip(measurements, sensor_angles):
            self.inverse_sensor_model(pose, z, angle, max_range)

    #Convert log-odds back to probability for visualization
    def get_probability_map(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))





    def draw_mapped_walls(self, threshold=0.8, color='orange', markersize=2):
        """Plot only the occupied cells as points — the walls the robot discovered."""
        prob = self.get_probability_map()
        occupied = np.argwhere(prob > threshold)  # [row, col] = [gy, gx]

        if len(occupied) == 0:
            return

        # Convert grid coords to world coords
        wx = occupied[:, 1] * self.resolution + self.x_min + self.resolution / 2
        wy = occupied[:, 0] * self.resolution + self.y_min + self.resolution / 2

        plt.plot(wx, wy, '.', color=color, markersize=markersize)