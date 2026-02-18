import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

walls = np.loadtxt("walls.txt")

for x1, y1, x2, y2 in walls:
    plt.plot([x1, x2], [y1, y2], 'k-')

plt.gca().set_aspect('equal')
plt.title("Floor Plan")

def draw_goal(position, size=0.3, color='g'):
    x, y = position
    square = plt.Rectangle((x - size/2, y - size/2), size, size, color=color)
    plt.gca().add_patch(square)

goal = np.array([5.0, 4.0])

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





def pz(z, z_true, sigma=0.5, lam=1.0, z_max=20.0, eps=0.1,
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

# we made pz based on the noise of real world, this just adds noise
def noisy_measure(pose, walls, angle_offset=0, max_range=20,
                  sigma=0.5, lam=1.0,
                  w_hit=0.70, w_short=0.10, w_max=0.10, w_rand=0.10):
    z_true = measure(pose, walls, angle_offset, max_range)

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
            #likelihood = np.exp(-(error ** 2) / (2 * sigma ** 2))
            likelihood = pz(z, predicted, sigma=sigma)
            weights[i] *= likelihood

    # Avoid division by zero
    weights += 1e-300

    # Normalize
    weights /= np.sum(weights)

    return weights

# 4. Normalization
# goes in algo
# 5. Resampling
def resample(particles, weights, walls, replace_ratio=0.0005):
    N = len(particles)

    # If weights are too uniform (no good matches), inject random particles
    n_eff = 1.0 / np.sum(weights**2)  # Effective sample size
    n_random = int(N * replace_ratio)

    # Resample based on weights
    indices = np.random.choice(N, size=N, p=weights)
    particles = particles[indices]

    # Replace some particles with random ones to avoid particle deprivation
    if n_eff < N * 0.1:  # If effective particles < 10% of total
        n_random = int(N * 0.2)  # Replace 20% instead

    # Get map bounds from walls
    x_min = min(w[0] for w in walls)
    x_max = max(w[2] for w in walls)
    y_min = min(w[1] for w in walls)
    y_max = max(w[3] for w in walls)

    # Replace last n_random particles with random poses
    random_indices = np.random.choice(N, size=n_random, replace=False)
    for i in random_indices:
        particles[i, 0] = np.random.uniform(x_min, x_max)
        particles[i, 1] = np.random.uniform(y_min, y_max)
        particles[i, 2] = np.random.uniform(-np.pi, np.pi)

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

# 7. Localization detection
def is_localized(particles, pos_threshold=0.5, angle_threshold=0.3):
    x_var = np.var(particles[:, 0])
    y_var = np.var(particles[:, 1])

    # Circular variance for angle
    cos_mean = np.mean(np.cos(particles[:, 2]))
    sin_mean = np.mean(np.sin(particles[:, 2]))
    angle_var = 1 - np.sqrt(cos_mean**2 + sin_mean**2)

    return (x_var < pos_threshold and
            y_var < pos_threshold and
            angle_var < angle_threshold)

# implementation of particle filter
N = 500
particles, weights = initialize_particles(N)
true_pose = pose

sensor_angles = [0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2]
plt.ion()

localized = 0
localized_step = None
localizing_limt = 5

for step in range(200):

    # Check if localized
    if is_localized(particles):
        localized += 1
        localized_step = step
        print(f"Localized at step {step}!")
        # if close turn and check
        if localized > 3:
            omega = np.random.uniform(0.3, 1.8)
            control = wheelspeed(-omega, omega, 1, 2)
        if localized > localizing_limt:
            break
    else:
        localized = 0

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
    v_noise = np.random.normal(0, 0.2)
    omega_noise = np.random.normal(0, 0.3)

    true_pose[0] += (control[0] + v_noise) * np.cos(true_pose[2])
    true_pose[1] += (control[0] + v_noise) * np.sin(true_pose[2])
    true_pose[2] += (control[1] + omega_noise)
    true_pose[2] = (true_pose[2] + np.pi) % (2 * np.pi) - np.pi  # normalize angle

    # Sensor reading for particle filter (after moving)
    z = [noisy_measure(true_pose, walls, angle_offset=a) for a in sensor_angles]

    # Particle filter
    particles = motion_update(particles, control, noise_std=[0.1, 0.05])
    weights = measurement_update(particles, weights, z, sigma=0.8)
    particles, weights = resample(particles, weights,walls)
    est_robot = estimate_state(particles, weights)

    # Plot
    plt.clf()
    for x1, y1, x2, y2 in walls:
        plt.plot([x1, x2], [y1, y2], 'k-')
    plt.scatter(particles[:, 0], particles[:, 1], s=1, c='blue', label='Particles')
    draw_robot(true_pose, size=0.3)          # red - true
    draw_robot(est_robot, size=0.3,color="orange")                # est - orange
    draw_sensors(true_pose, walls, sensor_angles)
    draw_goal(goal)
    plt.xlim(-1, 11)
    plt.ylim(-1, 9)
    plt.gca().set_aspect('equal')

    if localized > localizing_limt:
        plt.title(f"Step {step + 1}/200 — LOCALIZED (since step {localized_step})")
    else:
        plt.title(f"Step {step + 1}/200 — Localizing...")

    plt.pause(0.3)
plt.ioff()
plt.show()

#---------------------------











