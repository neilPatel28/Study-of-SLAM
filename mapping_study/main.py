import matplotlib.pyplot as plt
import numpy as np

from map import Map
from robot import Robot
from sensor import Sensor
from mappping import OccupancyGrid

# Setup
m = Map("walls.txt")
start_pose = [m.goal[0], m.goal[1], 0.0]  # start at green box
robot = Robot(start_pose=start_pose)
sensor = Sensor(m.walls, sensor_angles=[i * np.pi/6 for i in range(-6, 7)])

# Occupancy grid
og = OccupancyGrid(
    x_min=m.x_min, x_max=m.x_max,
    y_min=m.y_min, y_max=m.y_max,
    resolution=0.1
)

plt.ion()

# === Exploration parameters ===
explore_steps = 0

for step in range(100):

    # sensors readings
    z_front = sensor.measure(robot.pose)
    measurements = [sensor.noisy_measure(robot.pose, angle_offset=a) for a in sensor.sensor_angles]

    # update occupancy grid
    og.update(robot.pose, measurements, sensor.sensor_angles)

    # --- Decide control based on state ---
    explore_steps += 1

    # basic go forward and if wall then turn right,
    # better search algorithms can be added here
    if z_front < 0.8:
        control = robot.wheelspeed(0.3, -0.3, 1, 2)
    else:
        control = robot.wheelspeed(0.3, 0.3, 1, 2)

    if step > 500:
        break

    # NOW move true robot
    """v_noise = np.random.normal(0, 0.2)
    omega_noise = np.random.normal(0, 0.3)"""
    v_noise = 0
    omega_noise = 0

    robot.pose[0] += (control[0] + v_noise) * np.cos(robot.pose[2])
    robot.pose[1] += (control[0] + v_noise) * np.sin(robot.pose[2])
    robot.pose[2] += (control[1] + omega_noise)
    robot.pose[2] = (robot.pose[2] + np.pi) % (2 * np.pi) - np.pi  # normalize angle

    # --- Plot ---
    m.setup_frame()
    m.draw_walls()
    robot.draw_robot()
    sensor.draw_sensors(robot.pose)
    m.draw_goal()

    plt.title(f"Step {step+1} ")


    plt.pause(0.05)

# === Final map ===
plt.ioff()
plt.figure()
m.draw_walls()              # real world in black
og.draw_mapped_walls()      # mapped walls in orange on top
robot.draw_robot()
plt.xlim(m.x_min - m.margin, m.x_max + m.margin)
plt.ylim(m.y_min - m.margin, m.y_max + m.margin)
plt.gca().set_aspect('equal')
plt.title("Mapped vs Real Environment")
plt.show()