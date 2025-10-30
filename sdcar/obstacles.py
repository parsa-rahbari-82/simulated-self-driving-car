import random
from math import sqrt


def randomise_obstacle_positions(
    n_obs,
    arena_size,
    start_xy,
    goal_xy,
    min_start_dist = 1.5,
    min_goal_dist = 1.5,
    min_obs_dist = 1.0,
    max_retries = 1000,
):
    placed = []
    positions = [[0.0, 0.0, 0.0] for _ in range(n_obs)]
    for i in range(n_obs):
        retries = 0
        while retries < max_retries:
            x = random.uniform(1.0, arena_size - 1.0)
            y = random.uniform(1.0, arena_size - 1.0)
            z = 0.5
            dist_start = sqrt((x - start_xy[0]) ** 2 + (y - start_xy[1]) ** 2)
            dist_goal = sqrt((x - goal_xy[0]) ** 2 + (y - goal_xy[1]) ** 2)
            if dist_start <= min_start_dist or dist_goal <= min_goal_dist:
                retries += 1
                continue
            overlaps = False
            for prev_x, prev_y, _ in placed:
                if sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2) <= min_obs_dist:
                    overlaps = True
                    break
            if not overlaps:
                positions[i] = [x, y, z]
                placed.append([x, y, z])
                break
            retries += 1
        if retries >= max_retries:
            positions[i] = [1.0 + i, 1.0 + i, 0.5]
            print(
                f"Warning: could not place obstacle {i} without overlap."
                " Using a default position."
            )
    return positions

