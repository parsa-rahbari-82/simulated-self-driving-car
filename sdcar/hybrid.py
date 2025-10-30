from enum import Enum, auto
from dataclasses import dataclass
from math import atan2, pi
import numpy as np
import pybullet as p

from .control import track_u_with_unicycle, differential_drive_control


class Mode(Enum):
    GTG = auto()
    AO = auto()


@dataclass
class HybridParams:
    obstacle_radius: float = 0.5
    car_radius: float = 0.60
    clearance: float = 0.30

    v_max_far: float = 100.0
    v_max_near: float = 20.0

    ao_gain: float = 10.0
    ao_clip: float = 25.0

    eps_goal: float = 1.0
    band_width: float = 0.15
    ao_enter_dot: float = 0.0
    los_angle_tol: float = np.deg2rad(30.0)

    ray_z: float = 0.25
    ray_span_deg: float = 120.0
    ray_count: int = 10
    ray_stop_margin: float = 0.15

    def delta(self) -> float:
        return self.obstacle_radius + self.car_radius + self.clearance


def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / (n + eps)


class HybridController:
    def __init__(self, goal_xy, obstacle_ids, params: HybridParams):
        self.goal = np.asarray(goal_xy[:2], float)
        self.obs_ids = obstacle_ids
        self.prm = params
        self.mode = Mode.GTG

    def _nearest_obstacle(self, pos_xy):
        nearest, min_d = None, 1e9
        for oid in self.obs_ids:
            (ox, oy, _), _ = p.getBasePositionAndOrientation(oid)
            d = np.hypot(pos_xy[0] - ox, pos_xy[1] - oy)
            if d < min_d:
                min_d, nearest = d, np.array([ox, oy])
        d_center = min_d
        d_clearance = d_center - self.prm.obstacle_radius - self.prm.car_radius
        return nearest, d_center, d_clearance

    def _ray_fan_min_hit(self, pos_xy, heading):
        z = self.prm.ray_z
        span = np.deg2rad(self.prm.ray_span_deg)
        k = max(1, int(self.prm.ray_count))
        ds = []
        for i in range(k):
            a = -span / 2 + span * (i / (k - 1) if k > 1 else 0.5)
            th = heading + a
            end = [pos_xy[0] + 500 * np.cos(th), pos_xy[1] + 500 * np.sin(th), z]
            start = [pos_xy[0], pos_xy[1], z]
            hit = p.rayTest(start, end)[0]
            if hit[0] != -1 and hit[2] < 0.9999:
                dist = hit[2] * np.linalg.norm(np.array(end) - np.array(start))
                ds.append(dist)
        return np.min(ds) if ds else np.inf

    def u_gtg(self, pos_xy):
        return _normalize(self.goal - pos_xy)

    def u_ao(self, pos_xy):
        o, _, d_cl = self._nearest_obstacle(pos_xy)
        if o is None:
            return np.zeros(2)
        away = _normalize(pos_xy - o)
        mag = self.prm.ao_gain * (1.0 / max(d_cl, 1e-3))
        return _normalize(away * mag) * min(mag, self.prm.ao_clip)

    def _enter_ao(self, pos_xy, heading):
        _, d_center, _ = self._nearest_obstacle(pos_xy)
        in_tube = d_center <= self.prm.delta()
        rhit = self._ray_fan_min_hit(pos_xy, heading)
        predictive = rhit < (self.prm.delta() + self.prm.ray_stop_margin)
        if not (in_tube or predictive):
            return False
        return np.dot(self.u_ao(pos_xy), self.u_gtg(pos_xy)) > self.prm.ao_enter_dot

    def _can_go_gtg(self, pos_xy, theta):
        start = [pos_xy[0], pos_xy[1], self.prm.ray_z]
        end = [self.goal[0], self.goal[1], self.prm.ray_z]
        hit = p.rayTest(start, end)[0]
        los_clear = hit[0] == -1 or hit[2] >= 0.9999

        _, d_center, _ = self._nearest_obstacle(pos_xy)
        not_deep = d_center > self.prm.delta() - 0.05

        return los_clear and not_deep

    def _sliding_blend(self, pos_xy, u_gtg, u_ao):
        o, _, _ = self._nearest_obstacle(pos_xy)
        if o is None:
            return u_gtg
        n_hat = _normalize(pos_xy - o)
        num = -np.dot(n_hat, u_gtg)
        den = np.dot(n_hat, (u_ao - u_gtg))
        alpha = 0.5 if abs(den) < 1e-6 else np.clip(num / den, 0.0, 1.0)
        return _normalize(alpha * u_ao + (1.0 - alpha) * u_gtg)

    def step(self, pos_xy: np.ndarray, theta, wheel_base, wheel_radius):
        if self.mode == Mode.GTG:
            if self._enter_ao(pos_xy, theta):
                self.mode = Mode.AO
        else:
            if self._can_go_gtg(pos_xy, theta):
                self.mode = Mode.GTG

        u_g = self.u_gtg(pos_xy)
        if self.mode == Mode.GTG:
            u_xy = u_g
            vmax = self.prm.v_max_far
        else:
            u_a = self.u_ao(pos_xy)
            o, d_center, _ = self._nearest_obstacle(pos_xy)
            b = d_center - self.prm.delta()
            n_hat = _normalize(pos_xy - o) if o is not None else np.zeros(2)
            inward_g = np.dot(n_hat, u_g) < 0.0
            outward_a = np.dot(n_hat, u_a) > 0.0
            near_guard = abs(b) <= self.prm.band_width
            if near_guard and inward_g and outward_a:
                u_xy = self._sliding_blend(pos_xy, u_g, u_a)
            else:
                t_hat = np.array([-n_hat[1], n_hat[0]]) if np.any(n_hat) else u_g
                if np.dot(t_hat, u_g) < 0.0:
                    t_hat = -t_hat
                u_xy = _normalize(0.6 * t_hat + 0.4 * u_g)
            vmax = self.prm.v_max_near * 0.6

        u_xy = u_xy * vmax
        v, omega = track_u_with_unicycle(theta, u_xy, vmax=vmax)
        v_l, v_r = differential_drive_control(v, omega, wheel_base, wheel_radius)
        return v_l, v_r, self.mode
