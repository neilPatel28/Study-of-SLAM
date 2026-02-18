import matplotlib.pyplot as plt
import numpy as np


class Map:
    def __init__(self, filename="walls.txt"):
        self.walls = np.loadtxt(filename)
        self.goal = np.array([5.0, 4.0])

        # Auto-fit bounds from walls
        all_x = np.concatenate([self.walls[:, 0], self.walls[:, 2]])
        all_y = np.concatenate([self.walls[:, 1], self.walls[:, 3]])
        self.x_min = np.min(all_x)
        self.x_max = np.max(all_x)
        self.y_min = np.min(all_y)
        self.y_max = np.max(all_y)
        self.margin = 1.0

    def draw_walls(self):
        for x1, y1, x2, y2 in self.walls:
            plt.plot([x1, x2], [y1, y2], 'k-')

    def draw_goal(self, size=0.3, color='g'):
        x, y = self.goal
        square = plt.Rectangle((x - size/2, y - size/2), size, size, color=color)
        plt.gca().add_patch(square)

    def setup_frame(self):
        plt.clf()
        plt.xlim(self.x_min - self.margin, self.x_max + self.margin)
        plt.ylim(self.y_min - self.margin, self.y_max + self.margin)
        plt.gca().set_aspect('equal')