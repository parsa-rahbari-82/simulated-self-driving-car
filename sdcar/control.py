import numpy as np


def compute_u(pos_xy, goal_xy, k_xy = 2, max_speed = 50.0):
    vec = k_xy * (goal_xy - pos_xy)
    norm = np.linalg.norm(vec)
    if norm > max_speed and norm > 0:
        vec = vec / norm * max_speed
    return vec


def track_u_with_unicycle(theta, u_xy, k_phi = 100.0, vmax = 10.0, omega_max=6.0):
    ux, uy = u_xy
    
    # Calculate the desired heading from the velocity vector
    desired_heading = np.arctan2(uy, ux) if np.any(u_xy != 0) else theta
    
    # Calculate the heading error, wrapping it to the range [-pi, pi]
    e = (desired_heading - theta + np.pi) % (2 * np.pi) - np.pi
    
    omega = np.clip(k_phi * e, -omega_max, omega_max)
    
    v = min(np.hypot(ux, uy), vmax)
    
    return v, omega

def differential_drive_control(v, omega, wheel_base, wheel_radius):
    v_l = (2.0 * v - omega * wheel_base) / (2.0 * wheel_radius)
    v_r = (2.0 * v + omega * wheel_base) / (2.0 * wheel_radius)
    return v_l, v_r

