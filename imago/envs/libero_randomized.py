"""Standard LIBERO with *generic* domain randomization for zero-shot IMAGO RL.

Protocol (to stay comparable with DIDO / Faster-WAM): train only on the
standard LIBERO four-suite tasks and evaluate LIBERO-Plus zero-shot. A
FastWAM/DIDO-level policy already solves about 98–99% of standard LIBERO, so
GRPO would see almost no mixed-outcome groups. We therefore add generic
randomization written here. It uses no LIBERO-Plus assets, splits or
perturbation generators, and papers must describe it as "generic domain
randomization".

Applied after every ``set_init_state`` inside the simulator subprocess, and
reset to the scene's nominal values first, so perturbations never accumulate:

* robot initial joint noise, ``N(0, joint_noise_std)`` rad;
* agentview camera position jitter (uniform ±``camera_pos_jitter`` m) and a
  small random rotation (±``camera_rot_jitter_deg``);
* global light intensity scaling, uniform in ``light_scale``.

Wiring: ``env.train.imago_randomization.enabled: true`` makes RLinf's
(patched) ``get_env_cls`` return :class:`RandomizedLiberoEnv`. It creates
:class:`RandomizedOffScreenRenderEnv` workers and passes ``_env_cls`` through
the params, so the worker's ``reconfigure`` command (patched) re-creates the
randomized class too.
"""

from __future__ import annotations

import os

import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from omegaconf import OmegaConf

from rlinf.envs.sim.libero.libero_env import LiberoEnv

DEFAULTS = {
    "camera": "agentview",
    "joint_noise_std": 0.05,
    "camera_pos_jitter": 0.02,
    "camera_rot_jitter_deg": 3.0,
    "light_scale": (0.6, 1.4),
}


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _small_rotation(rng, max_deg):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-8
    angle = np.deg2rad(rng.uniform(-max_deg, max_deg))
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


class RandomizedOffScreenRenderEnv(OffScreenRenderEnv):
    def __init__(self, imago_rand: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        params = dict(DEFAULTS)
        params.update(imago_rand or {})
        params["light_scale"] = tuple(params["light_scale"])
        self._rand = params
        self._rng = np.random.default_rng(0)
        self._nominal = None

    def seed(self, seed):
        out = super().seed(seed)
        self._rng = np.random.default_rng(int(seed) + 7919)
        return out

    def _capture_nominal(self, sim):
        model = sim.model
        cam = model.camera_name2id(self._rand["camera"])
        self._nominal = {
            "model_id": id(model),
            "cam": cam,
            "cam_pos": model.cam_pos[cam].copy(),
            "cam_quat": model.cam_quat[cam].copy(),
            "light_diffuse": model.light_diffuse.copy(),
            "light_specular": model.light_specular.copy(),
        }

    def randomize(self) -> None:
        sim = self.env.sim
        if self._nominal is None or self._nominal["model_id"] != id(sim.model):
            self._capture_nominal(sim)
        nom, p, rng, model = self._nominal, self._rand, self._rng, sim.model
        model.cam_pos[nom["cam"]] = nom["cam_pos"] + rng.uniform(-p["camera_pos_jitter"], p["camera_pos_jitter"], 3)
        model.cam_quat[nom["cam"]] = _quat_mul(nom["cam_quat"], _small_rotation(rng, p["camera_rot_jitter_deg"]))
        scale = rng.uniform(*p["light_scale"])
        model.light_diffuse[:] = np.clip(nom["light_diffuse"] * scale, 0, 1)
        model.light_specular[:] = np.clip(nom["light_specular"] * scale, 0, 1)
        if p["joint_noise_std"] > 0:
            ids = self.env.robots[0]._ref_joint_pos_indexes
            sim.data.qpos[ids] = sim.data.qpos[ids] + rng.normal(0, p["joint_noise_std"], size=len(ids))
        sim.forward()

    def set_init_state(self, init_state):
        obs = super().set_init_state(init_state)
        self.randomize()
        return obs  # LiberoEnv runs settle steps before observing


class RandomizedLiberoEnv(LiberoEnv):
    """Standard-LIBERO training env whose workers are randomized."""

    def __init__(self, cfg, *args, **kwargs):
        variant = os.environ.get("LIBERO_TYPE", cfg.get("libero_variant", "standard"))
        if variant != "standard":
            raise ValueError(
                "imago_randomization is for zero-shot training on standard LIBERO only "
                f"(LIBERO_TYPE={variant!r}); evaluate LIBERO-Plus in a separate run without it."
            )
        super().__init__(cfg, *args, **kwargs)

    def _rand_params(self) -> dict:
        params = OmegaConf.to_container(self.cfg.imago_randomization, resolve=True)
        params.pop("enabled", None)
        return params

    def get_env_fn_params(self, env_idx=None):
        params = super().get_env_fn_params(env_idx)
        for p in params:
            p["_env_cls"] = RandomizedOffScreenRenderEnv  # consumed by the patched "reconfigure"
            p["imago_rand"] = self._rand_params()
        return params

    def get_env_fns(self):
        fns = []
        for param in self.get_env_fn_params():
            def env_fn(param=param):
                os.environ["LIBERO_TYPE"] = "standard"
                param = dict(param)
                seed = param.pop("seed")
                cls = param.pop("_env_cls")
                env = cls(**param)
                env.seed(seed)
                return env
            fns.append(env_fn)
        return fns
