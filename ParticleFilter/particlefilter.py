import matplotlib.pyplot as plt
import numpy as np

walls = np.loadtxt("walls.txt")

for x1, y1, x2, y2 in walls:
    plt.plot([x1, x2], [y1, y2], 'k-')

plt.gca().set_aspect('equal')
plt.title("Floor Plan")

def draw_robot(pose, size=0.3,color='r'):
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

# Random pose
pose = np.array([
    np.random.uniform(0+0.4, 10-0.4),
    np.random.uniform(0+0.4, 8-0.4),
    np.random.uniform(-np.pi, np.pi)
])

"""# Draw robot
draw_robot(pose)
plt.show()"""



#----------------------------------------
#filter
# simulate a sensor
def sensor(ray_origin, ray_dir, p1, p2):
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

def measure(pose, walls, angle_offset=0, max_range=20):
    x, y, theta = pose
    ray_origin = np.array([x, y])
    ray_dir = np.array([np.cos(theta + angle_offset), np.sin(theta + angle_offset)])
    min_dist = max_range

    for x1, y1, x2, y2 in walls:
        dist = sensor(ray_origin, ray_dir, np.array([x1, y1]), np.array([x2, y2]))

        if dist is not None and dist < min_dist:
            min_dist = dist

    return min_dist


def draw_sensors(pose, walls, sensor_angles, max_range=20):
    x, y, theta = pose
    for angle in sensor_angles:
        dist = measure(pose, walls, angle_offset=angle, max_range=max_range)
        end_x = x + dist * np.cos(theta + angle)
        end_y = y + dist * np.sin(theta + angle)
        plt.plot([x, end_x], [y, end_y], 'g-', alpha=0.5)
        plt.plot(end_x, end_y, 'go', markersize=3)


# map wheel speeds to linear and angular velocity (v,w)
def wheelspeed(phi_l, phi_r, r=1, w=2):
    v = r / 2 * (phi_r + phi_l)
    omega = r / w * (phi_r - phi_l)
    return v, omega


# 1. Initialization (at k = 0)
def initialize_particles(N):
    particles = np.zeros((N, 3))

    # Sample from prior (uniform over map)
    particles[:, 0] = np.random.uniform(0, 10, N)      # x
    particles[:, 1] = np.random.uniform(0, 8, N)       # y
    particles[:, 2] = np.random.uniform(-np.pi, np.pi, N)  # theta

    weights = np.ones(N) / N  # equal weight, unfiform proabilty of being anywhere

    return particles, weights

#2. Prediction (motion update)
def motion_update(particles, control, noise_std):
    v, omega = control

    N = len(particles)

    # Add noise to control
    v_noisy = v + np.random.normal(0, noise_std[0], N)
    omega_noisy = omega + np.random.normal(0, noise_std[1], N)

    # Update state apply map to output to (x,y,theta)
    particles[:, 0] += v_noisy * np.cos(particles[:, 2])
    particles[:, 1] += v_noisy * np.sin(particles[:, 2])
    particles[:, 2] += omega_noisy

    return particles

# 3. Measurement update (weighting)
def measurement_update(particles, weights, measurements, sigma):
    for i, p in enumerate(particles):
        for angle, z in zip(sensor_angles, measurements):
            predicted = measure(p, walls, angle_offset=angle)
            error = z - predicted
            likelihood = np.exp(-(error ** 2) / (2 * sigma ** 2))
            weights[i] *= likelihood

    # Avoid division by zero
    weights += 1e-300

    # Normalize
    weights /= np.sum(weights)

    return weights

# 4. Normalization
# goes in algo
# 5. Resampling
def resample(particles, weights):
    N = len(particles)

    # this numpy packaage grabs particles based on weights, getting back N points
    indices = np.random.choice(
        N,
        size=N,
        p=weights
    )
    # does the action
    particles = particles[indices]

    #now from the good options, reset
    weights = np.ones(N) / N

    return particles, weights

# 6. State estimation (from the particles right now, where is the robot)
# just for us to see where thing robots
def estimate_state(particles, weights):
    #expection value for x,y,theta
    x = np.average(particles[:, 0], weights=weights)
    y = np.average(particles[:, 1], weights=weights)

    cos_sum = np.average(np.cos(particles[:, 2]), weights=weights)
    sin_sum = np.average(np.sin(particles[:, 2]), weights=weights)
    theta = np.arctan2(sin_sum, cos_sum)

    return np.array([x, y, theta])

# implementation of particle filter
N = 500
particles, weights = initialize_particles(N)
true_pose = pose

sensor_angles = [0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2]
plt.ion()

for step in range(200):

    # Check distance BEFORE moving
    z_front = measure(true_pose, walls)
    #z_front = z[0]

    if z_front < 1:
        # Wall is close — stop and turn
        omega = np.random.uniform(0.3, 1.8)
        control = wheelspeed(-omega, omega, 1, 2)

    else:
        # Clear ahead — move forward
        # this maps to (3,0) in v,w
        control = wheelspeed(0.3, 0.3, 1, 2)

    # NOW move true robot
    true_pose[0] += control[0] * np.cos(true_pose[2])
    true_pose[1] += control[0] * np.sin(true_pose[2])
    true_pose[2] += control[1]
    true_pose[2] = (true_pose[2] + np.pi) % (2 * np.pi) - np.pi # normailze angle

    # Sensor reading for particle filter (after moving)
    z = [measure(true_pose, walls, angle_offset=a) for a in sensor_angles]

    # Particle filter
    particles = motion_update(particles, control, noise_std=[0.1, 0.05])
    weights = measurement_update(particles, weights, z, sigma=0.8)
    particles, weights = resample(particles, weights)
    est_robot = estimate_state(particles, weights)

    # Plot
    plt.clf()
    for x1, y1, x2, y2 in walls:
        plt.plot([x1, x2], [y1, y2], 'k-')
    plt.scatter(particles[:, 0], particles[:, 1], s=1, c='blue', label='Particles')
    draw_robot(true_pose, size=0.3)          # red - true
    draw_robot(est_robot, size=0.3,color="orange")                # est - orange
    draw_sensors(true_pose, walls, sensor_angles)
    plt.xlim(-1, 11)
    plt.ylim(-1, 9)
    plt.gca().set_aspect('equal')
    plt.title(f"Step {step + 1}/200")
    plt.pause(0.3)

plt.ioff()
plt.show()








# i want to make racecar given a racetrack to be better than my friend driving a rc car
# in a maze, can we make robot faster than human?



# bassilly we start a robot and goes through a racetrack - simulate
"""
Things to add
 - other cars
 - best paths
 - player vs robot
 - real life
 - 4 wheels 
 - change start to start line
 - machine learning like bump the car? can we teach it tech?
            - because its robot it can control all wheels for path, unlike humans stuck in real wheel or full
 - skidding
class
 - other cars
 - skidding
 - best angles for camera and best car desgin?
 - slam or local?
"""