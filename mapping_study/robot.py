import matplotlib.pyplot as plt
import numpy as np


class Robot:
    def __init__(self, start_pose=None):
        if start_pose is not None:
            self.pose = np.array(start_pose, dtype=float)
        else:
            self.pose = np.array([
                np.random.uniform(0+0.4, 10-0.4),
                np.random.uniform(0+0.4, 8-0.4),
                np.random.uniform(-np.pi, np.pi)
            ])

    def draw_robot(self, pose=None, size=0.3, color='r'):
        if pose is None:
            pose = self.pose
        x, y, theta = pose

        # Triangle in robot local frame
        triangle = np.array([
            [ size,  0],          # front tip
            [-size, -size/2],     # back left
            [-size,  size/2]      # back right
        ])

        # Rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # Rotate and translate
        triangle_world = (R @ triangle.T).T
        triangle_world += np.array([x, y])

        # Close the triangle
        triangle_world = np.vstack([triangle_world, triangle_world[0]])

        # Plot
        plt.plot(triangle_world[:,0], triangle_world[:,1], color=color)

    # map wheel speeds to linear and angular velocity (v,w)
    def wheelspeed(self, phi_l, phi_r, r=1, w=2):
        v = r / 2 * (phi_r + phi_l)
        omega = r / w * (phi_r - phi_l)
        return v, omega