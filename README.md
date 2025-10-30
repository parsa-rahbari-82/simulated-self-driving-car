# Simulated Self-Driving Car (PyBullet)

A minimal self-driving car sandbox using PyBullet. It spawns a differential-drive robot in a 3D arena with random spherical obstacles and drives to a goal using a hybrid controller: go-to-goal (GTG) with obstacle avoidance (AO), switching modes based on simple geometric/line-of-sight cues.

**Features**

- PyBullet physics with simple chassis + 4 wheel links
- Random obstacle field and visual goal marker
- Hybrid GTG/AO controller with ray checks and nearest-obstacle logic
- Unicycle tracking + differential drive conversion
- Configurable world, robot, and control parameters

**Quick Start**

1) Create a virtual environment and install deps:
   - python -m venv venv
   - venv\Scripts\activate (Windows) or source venv/bin/activate (macOS/Linux)
   - pip install -r requirements.txt (or pip install pybullet numpy)
2) Run the simulation with GUI:
   - python main.py

To run headless, import and call sdcar.simulation.run_simulation(show_gui=False).

**Project Structure**

- main.py — entry point; runs the simulation with GUI.
- sdcar/config.py — simulation constants (arena size, timing, robot dims, forces).
- sdcar/control.py — low-level control: GTG velocity field, unicycle tracking, wheel speeds.
- sdcar/hybrid.py — hybrid controller (GTG/AO) and parameters.
- sdcar/simulation.py — world setup, obstacle randomization, PyBullet loop, termination.

**Key Parameters**
Adjust in sdcar/config.py and sdcar/hybrid.py:

- World: NUM_OBSTACLES, ARENA_SIZE, START_POS, GOAL_POS, DT, T_SIM
- Robot: WHEEL_RADIUS, WHEEL_BASE, MAX_FORCE
- Hybrid control: HybridParams (radii/clearance, ray settings, speed caps, goal epsilon)

**How It Works**

- The controller computes a desired planar velocity toward the goal.
- Rays and nearest-obstacle checks decide when to switch to AO mode.
- Unicycle tracking maps the planar velocity to (v, omega); wheel speeds are commanded in PyBullet.
- The loop steps physics, checks for goal arrival or collisions, and stops.

**Notes**

> Requires a display for GUI mode. Use show_gui=False for headless runs.

> Tuning HybridParams can greatly affect smoothness and safety margins.
