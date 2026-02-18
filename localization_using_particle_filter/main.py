import matplotlib.pyplot as plt
import numpy as np

from map import Map
from robot import Robot
from sensor import Sensor
from particlefilter import ParticleFilter

# Setup
m = Map("racetrack.txt")
robot = Robot()
sensor = Sensor(m.walls)
pf = ParticleFilter(500, sensor)

plt.ion()

localized = 0
localized_step = None
localizing_limt = 5

for step in range(200):

    # Check if localized
    if pf.is_localized():
        localized += 1
        localized_step = step
        print(f"Localized at step {step}!")
        # if close turn and check
        if localized > 3:
            omega = np.random.uniform(0.3, 1.8)
            control = robot.wheelspeed(-omega, omega, 1, 2)
        if localized > localizing_limt:
            break
    else:
        localized = 0

    # Check distance BEFORE moving
    z_front = sensor.measure(robot.pose)

    if z_front < 1:
        # Wall is close — stop and turn
        omega = np.random.uniform(0.3, 1.8)
        control = robot.wheelspeed(-omega, omega, 1, 2)
    else:
        # Clear ahead — move forward
        # this maps to (3,0) in v,w
        control = robot.wheelspeed(0.3, 0.3, 1, 2)

    # NOW move true robot
    v_noise = np.random.normal(0, 0.2)
    omega_noise = np.random.normal(0, 0.3)

    robot.pose[0] += (control[0] + v_noise) * np.cos(robot.pose[2])
    robot.pose[1] += (control[0] + v_noise) * np.sin(robot.pose[2])
    robot.pose[2] += (control[1] + omega_noise)
    robot.pose[2] = (robot.pose[2] + np.pi) % (2 * np.pi) - np.pi  # normalize angle

    # Sensor reading for particle filter (after moving)
    z = [sensor.noisy_measure(robot.pose, angle_offset=a) for a in sensor.sensor_angles]

    # Particle filter
    pf.motion_update(control, noise_std=[0.1, 0.05])
    pf.measurement_update(z, sigma=0.8)
    pf.resample()
    est_robot = pf.estimate_state()

    # Plot
    m.setup_frame()
    m.draw_walls()
    plt.scatter(pf.particles[:, 0], pf.particles[:, 1], s=1, c='blue', label='Particles')
    robot.draw_robot()                                   # red - true
    robot.draw_robot(est_robot, size=0.3, color="orange")  # est - orange
    sensor.draw_sensors(robot.pose)
    m.draw_goal()

    if localized > localizing_limt:
        plt.title(f"Step {step + 1}/200 — LOCALIZED (since step {localized_step})")
    else:
        plt.title(f"Step {step + 1}/200 — Localizing...")

    plt.pause(0.3)

plt.ioff()
plt.show()