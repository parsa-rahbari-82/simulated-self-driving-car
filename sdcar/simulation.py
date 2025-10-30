from math import sqrt
from time import sleep

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data

from .config import (
    ARENA_SIZE,
    START_POS,
    GOAL_POS,
    DT,
    T_SIM,
    GOAL_THRESHOLD,
    SLOWDOWN_RADIUS,
    WHEEL_BASE,
    WHEEL_RADIUS,
    MAX_FORCE,
    NUM_OBSTACLES,
)
from .control import compute_u, track_u_with_unicycle, differential_drive_control
from .obstacles import randomise_obstacle_positions
from .hybrid import HybridController, HybridParams, Mode


def setup_simulation(
    start_pos,
    obstacle_positions,
    goal_pos,
    wheel_radius = WHEEL_RADIUS,
    wheel_base = WHEEL_BASE,
    show_gui= True,
):
    physics_client = p.GUI if show_gui else p.DIRECT
    p.connect(physics_client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(0)

    if show_gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=10.0,
            cameraYaw=90.0,
            cameraPitch=-45.0,
            cameraTargetPosition=[ARENA_SIZE / 2.0, ARENA_SIZE / 2.0, 0.0],
        )

    p.loadURDF("plane.urdf", [0, 0, 0])

    chassis_half_extents = [0.5, 0.3, 0.1]
    chassis_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=chassis_half_extents)
    chassis_vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=chassis_half_extents, rgbaColor=[0.0, 0.0, 1.0, 1.0]
    )
    chassis_start_ori = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

    wheel_length = 0.05
    wheel_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=wheel_radius, height=wheel_length)
    wheel_vis = p.createVisualShape(
        p.GEOM_CYLINDER, radius=wheel_radius, length=wheel_length, rgbaColor=[0.5, 0.5, 0.5, 1.0]
    )
    wheel_positions = [
        [-wheel_base / 2.0, 0.3, 0.1],   # rear left (0)
        [-wheel_base / 2.0, -0.3, 0.1],  # rear right (1)
        [wheel_base / 2.0, 0.3, 0.1],    # front left (2)
        [wheel_base / 2.0, -0.3, 0.1],   # front right (3)
    ]
    wheel_ori_left = p.getQuaternionFromEuler([np.pi / 2.0, 0.0, 0.0])
    wheel_ori_right = p.getQuaternionFromEuler([-np.pi / 2.0, 0.0, 0.0])
    wheel_orientations = [wheel_ori_left, wheel_ori_right, wheel_ori_left, wheel_ori_right]

    chassis_mass = 10.0
    wheel_mass = 1.0
    link_masses = [wheel_mass] * 4
    link_collision = [wheel_col] * 4
    link_visual = [wheel_vis] * 4
    link_positions = wheel_positions
    link_orientations = wheel_orientations
    link_inertial_pos = [[0.0, 0.0, 0.0]] * 4
    link_inertial_ori = [[0.0, 0.0, 0.0, 1.0]] * 4
    link_parent_indices = [0] * 4
    link_joint_types = [p.JOINT_REVOLUTE] * 4
    link_joint_axes = [[0.0, 0.0, 1.0]] * 4

    car_id = p.createMultiBody(
        baseMass=chassis_mass,
        baseCollisionShapeIndex=chassis_col,
        baseVisualShapeIndex=chassis_vis,
        basePosition=start_pos,
        baseOrientation=chassis_start_ori,
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_collision,
        linkVisualShapeIndices=link_visual,
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkInertialFramePositions=link_inertial_pos,
        linkInertialFrameOrientations=link_inertial_ori,
        linkParentIndices=link_parent_indices,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axes,
    )

    p.changeDynamics(car_id, -1, lateralFriction=1.0)
    for i in range(4):
        p.changeDynamics(
            car_id, i, lateralFriction=1.5, spinningFriction=0.01, rollingFriction=0.01
        )

    n_joints = p.getNumJoints(car_id)
    for j in range(n_joints):
        p.setJointMotorControl2(car_id, j, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=0)

    obstacle_ids = []
    for pos in obstacle_positions:
        obs_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
        obs_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[1.0, 0.0, 0.0, 1.0])
        obs_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=obs_col,
            baseVisualShapeIndex=obs_vis,
            basePosition=pos,
        )
        obstacle_ids.append(obs_id)

    goal_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.4)
    goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.4, rgbaColor=[0.0, 1.0, 0.0, 1.0])
    goal_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=goal_col,
        baseVisualShapeIndex=goal_vis,
        basePosition=goal_pos,
    )

    return car_id, obstacle_ids, goal_id


def apply_differential_drive(car_id, v_l, v_r):
    p.setJointMotorControl2(car_id, 2, controlMode=p.VELOCITY_CONTROL, targetVelocity=-v_l, force=MAX_FORCE)
    p.setJointMotorControl2(car_id, 0, controlMode=p.VELOCITY_CONTROL, targetVelocity=-v_l, force=MAX_FORCE) 
    p.setJointMotorControl2(car_id, 1, controlMode=p.VELOCITY_CONTROL, targetVelocity=v_r, force=MAX_FORCE)  
    p.setJointMotorControl2(car_id, 3, controlMode=p.VELOCITY_CONTROL, targetVelocity=v_r, force=MAX_FORCE)


def run_simulation(show_gui=True):
    obstacle_positions = randomise_obstacle_positions(
        NUM_OBSTACLES,
        ARENA_SIZE,
        (START_POS[0], START_POS[1]),
        (GOAL_POS[0], GOAL_POS[1]),
    )

    car_id, obstacle_ids, goal_id = setup_simulation(
        start_pos=START_POS,
        obstacle_positions=obstacle_positions,
        goal_pos=GOAL_POS,
        wheel_radius=WHEEL_RADIUS,
        wheel_base=WHEEL_BASE,
        show_gui=show_gui,
    )

    steps = int(T_SIM / DT)
    poses = []
    
    prm = HybridParams(
        obstacle_radius=0.5,
        car_radius=0.60, 
        clearance=0.4,
        v_max_far=4.0,
        v_max_near=2.0,
    )
    ctrl = HybridController(goal_xy=np.array(GOAL_POS[:2]),
                            obstacle_ids=obstacle_ids,
                            params=prm)
    for i in range(steps):
        pos, ori = p.getBasePositionAndOrientation(car_id)
        pos_xy = np.array([pos[0], pos[1]])
        theta = p.getEulerFromQuaternion(ori)[2]

        dist_to_goal = sqrt((pos[0]-GOAL_POS[0])**2 + (pos[1]-GOAL_POS[1])**2)
        if dist_to_goal < prm.eps_goal:
            apply_differential_drive(car_id, 0.0, 0.0)
            p.stepSimulation()
            print(f"Reached goal at step {i}! d={dist_to_goal:.2f}")
            break

        v_l, v_r, mode = ctrl.step(pos_xy, theta, WHEEL_BASE, WHEEL_RADIUS)
        apply_differential_drive(car_id, v_l, v_r)
        p.stepSimulation()

        poses.append([pos[0], pos[1], theta])

        collided = False
        for obs_id in obstacle_ids:
            if p.getContactPoints(bodyA=car_id, bodyB=obs_id):
                collided = True
                break
        if collided:
            print(f"Collision at step {i}! Terminating simulation.")
            break

        if show_gui:
            sleep(DT)

    p.disconnect()

    if poses:
        poses_arr = np.array(poses)
        plt.figure(figsize=(8, 6))
        plt.plot(poses_arr[:, 0], poses_arr[:, 1], '-', label='Trajectory')
        plt.plot(START_POS[0], START_POS[1], 'go', label='Start')
        plt.plot(GOAL_POS[0], GOAL_POS[1], 'ro', label='Goal')
        obs_x = [pos[0] for pos in obstacle_positions]
        obs_y = [pos[1] for pos in obstacle_positions]
        plt.plot(obs_x, obs_y, 'rs', label='Obstacles')
        plt.axis('equal')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Car trajectory with PID control')
        plt.grid(True)
        plt.legend()
        plt.show()

        final_dist = sqrt((poses_arr[-1, 0] - GOAL_POS[0]) ** 2 + (poses_arr[-1, 1] - GOAL_POS[1]) ** 2)
        print(f"Final distance to goal: {final_dist:.2f} m | Steps taken: {len(poses_arr)}")
    else:
        print("No trajectory data logged.")
