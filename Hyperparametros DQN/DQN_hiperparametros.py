from __future__ import annotations

import os
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
NOMBRE_ARCHIVO = "Viernes.csv"
SEED = 42

FIXED_START_STEP = 0
FORCE_CPU = False

# Entorno / Reward
TARGET_TEMP = 23.0
COMFORT_RANGE = 1.5
W_E = 0.6
W_C = 0.4
POWER_ACTION_SCALE = 0.6

# Grid Search
GRID_TRAIN_EPISODES = 60
EVAL_EPISODES = 5          #  antes era 1. 
EVAL_SEED_BASE = 1000      # para hacer eval reproducible pero variando ventana
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, NOMBRE_ARCHIVO)
OUT_DIR = BASE_DIR
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# SEED
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# =========================================================
# DATA LOAD
# =========================================================
DEFAULT_COL_MAP = {
    "Act P (kW)": "TAP",
    "Act P(kW)": "TAP",
    "TAP": "TAP",

    "TempR1 (C)": "TF1",
    "TempR2 (C)": "TF2",
    "TempR3 (C)": "TF3",
    "TF1": "TF1",
    "TF2": "TF2",
    "TF3": "TF3",

    "HUMEDADR1 (%rH)": "HF1",
    "HF1": "HF1",

    "Local Time Stamp": "Timestamp",
    "Timestamp": "Timestamp",
}

REQUIRED_COLS = ["Timestamp", "TAP", "TF1", "TF2", "TF3", "HF1"]


def load_and_prepare(csv_path: str, col_map: Dict[str, str] = DEFAULT_COL_MAP) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No existe el archivo: {csv_path}")

    df_raw = pd.read_csv(csv_path)

    rename_dict = {c: col_map[c] for c in df_raw.columns if c in col_map}
    df = df_raw.rename(columns=rename_dict).copy()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas {missing}. Columnas disponibles: {list(df.columns)}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).reset_index(drop=True)

    hour = df["Timestamp"].dt.hour.astype(int)
    minute = df["Timestamp"].dt.minute.astype(int)
    time_mins = hour * 60 + minute
    df["hour_sin"] = np.sin(2 * np.pi * time_mins / (24 * 60))
    df["hour_cos"] = np.cos(2 * np.pi * time_mins / (24 * 60))

    df = df.ffill().bfill()

    for c in ["TAP", "TF1", "TF2", "TF3", "HF1", "hour_sin", "hour_cos"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.ffill().bfill()

    return df


# =========================================================
# ENV
# =========================================================
class BuildingEnergyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        target_temp: float = 23.0,
        comfort_range: float = 1.5,
        w_e: float = 0.6,
        w_c: float = 0.4,
        power_action_scale: float = 0.6,
        seed: int = 42,
        fixed_start_step: Optional[int] = None,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.rng = np.random.default_rng(seed)
        self.fixed_start_step = fixed_start_step

        if len(self.df) < 3:
            raise ValueError("Dataset muy pequeño para detectar frecuencia.")
        time_delta = self.df["Timestamp"].iloc[1] - self.df["Timestamp"].iloc[0]
        minutes = float(time_delta.total_seconds() / 60.0)

        if 13 <= minutes <= 17:
            self.episode_length = 96
            self.freq_name = "15 minutos"
            self.temp_inertia = 1.5
            self.minutes_per_step = 15
        elif 55 <= minutes <= 65:
            self.episode_length = 24
            self.freq_name = "1 Hora"
            self.temp_inertia = 2.5
            self.minutes_per_step = 60
        else:
            self.episode_length = 96
            self.freq_name = f"Desconocida ({minutes:.1f} min) -> default 15min"
            self.temp_inertia = 1.5
            self.minutes_per_step = 15

        self.target_temp = float(target_temp)
        self.comfort_range = float(comfort_range)
        self.w_e = float(w_e)
        self.w_c = float(w_c)
        self.power_action_scale = float(power_action_scale)

        self.max_power = float(self.df["TAP"].max()) if float(self.df["TAP"].max()) > 0 else 1.0

        # DQN: acciones discretas
        self.action_bins = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_bins))

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.action_space.seed(seed)

        self.start_step = 0
        self.current_step_index = 0

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.action_space.seed(seed)

        max_idx = len(self.df) - self.episode_length - 2
        if max_idx <= 0:
            raise ValueError("Dataset no tiene suficientes filas para un episodio completo.")

        if self.fixed_start_step is not None:
            self.start_step = int(np.clip(self.fixed_start_step, 0, max_idx - 1))
        else:
            self.start_step = int(self.rng.integers(0, max_idx))

        self.current_step_index = 0
        return self._get_obs(self.start_step), {}

    def _get_obs(self, step_idx: int) -> np.ndarray:
        row = self.df.iloc[step_idx]
        return np.array(
            [row["TAP"], row["TF1"], row["TF2"], row["TF3"], row["HF1"], row["hour_sin"], row["hour_cos"]],
            dtype=np.float32,
        )

    def step(self, action: int):
        act_val = float(self.action_bins[int(action)])
        real_idx = self.start_step + self.current_step_index

        next_row = self.df.iloc[real_idx + 1]
        base_tap = float(next_row["TAP"])
        base_temp_avg = float((next_row["TF1"] + next_row["TF2"] + next_row["TF3"]) / 3.0)

        power_mod = 1.0 + (act_val * self.power_action_scale)
        temp_mod = -(act_val * self.temp_inertia)

        sim_tap = max(0.0, base_tap * power_mod)
        sim_temp = base_temp_avg + temp_mod

        # Reward
        energy_saving = (base_tap - sim_tap) / self.max_power
        dist = abs(sim_temp - self.target_temp)
        viol_norm = dist / self.comfort_range
        viol_clip = float(np.clip(viol_norm, 0.0, 1.0))

        reward = (self.w_e * energy_saving) - (self.w_c * viol_clip)

        self.current_step_index += 1
        terminated = self.current_step_index >= self.episode_length

        next_obs = self._get_obs(real_idx + 1)
        next_obs[0] = sim_tap
        next_obs[1] = next_obs[1] + temp_mod
        next_obs[2] = next_obs[2] + temp_mod
        next_obs[3] = next_obs[3] + temp_mod

        info = {"real_tap": base_tap, "dqn_tap": sim_tap, "sim_temp": sim_temp}
        return next_obs, float(reward), terminated, False, info


# =========================================================
# METRICS
# =========================================================
def compute_metrics(env: BuildingEnergyEnv, real_tap: np.ndarray, algo_tap: np.ndarray, temp: np.ndarray):
    kwh_real = real_tap.sum() * (env.minutes_per_step / 60.0)
    kwh_algo = algo_tap.sum() * (env.minutes_per_step / 60.0)

    savings_pct = ((kwh_real - kwh_algo) / max(kwh_real, 1e-6)) * 100.0

    low = env.target_temp - env.comfort_range
    high = env.target_temp + env.comfort_range
    viol = np.logical_or(temp < low, temp > high).astype(int)
    viol_pct = viol.mean() * 100.0

    rmse = math.sqrt(np.mean((temp - env.target_temp) ** 2))

    return {
        "kwh_real": float(kwh_real),
        "kwh_algo": float(kwh_algo),
        "savings_pct": float(savings_pct),
        "viol_pct": float(viol_pct),
        "rmse_temp": float(rmse),
    }


# =========================================================
# DQN
# =========================================================
class QNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    eps_decay: float = 0.995
    tau: float = 0.01
    batch_size: int = 64
    buffer_size: int = 50000
    start_learning: int = 200
    train_steps_per_ep: int = 1
    hidden: int = 256


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.state_dim = state_dim
        self.ptr = 0
        self.size = 0
        self.s = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity,), dtype=np.int64)
        self.r = np.zeros((self.capacity,), dtype=np.float32)
        self.ns = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((self.capacity,), dtype=np.float32)

    def add(self, s, a, r, ns, d):
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.ns[i] = ns
        self.d[i] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.ns[idx], self.d[idx]


def soft_update(target: nn.Module, online: nn.Module, tau: float):
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)


def train_dqn_short(env: BuildingEnergyEnv, cfg: DQNConfig, device: torch.device, episodes: int):
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    online = QNet(state_dim, n_actions, hidden=cfg.hidden).to(device)
    target = QNet(state_dim, n_actions, hidden=cfg.hidden).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    opt = optim.Adam(online.parameters(), lr=cfg.lr)
    buf = ReplayBuffer(cfg.buffer_size, state_dim)

    eps = 1.0
    eps_min = 0.05
    returns = []

    for ep in range(episodes):
        s, _ = env.reset(seed=SEED + ep)

        done = False
        ep_return = 0.0

        while not done:
            if np.random.rand() < eps:
                a = int(env.action_space.sample())
            else:
                st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q = online(st)
                a = int(torch.argmax(q, dim=1).item())

            ns, r, term, trunc, _info = env.step(a)
            done = term or trunc

            buf.add(s, a, r, ns, float(done))
            s = ns
            ep_return += float(r)

            if buf.size >= max(cfg.start_learning, cfg.batch_size):
                for _ in range(cfg.train_steps_per_ep):
                    bs, ba, br, bns, bd = buf.sample(cfg.batch_size)

                    bs_t = torch.tensor(bs, dtype=torch.float32, device=device)
                    ba_t = torch.tensor(ba, dtype=torch.int64, device=device).unsqueeze(1)
                    br_t = torch.tensor(br, dtype=torch.float32, device=device).unsqueeze(1)
                    bns_t = torch.tensor(bns, dtype=torch.float32, device=device)
                    bd_t = torch.tensor(bd, dtype=torch.float32, device=device).unsqueeze(1)

                    q_sa = online(bs_t).gather(1, ba_t)

                    with torch.no_grad():
                        next_a = torch.argmax(online(bns_t), dim=1, keepdim=True)
                        q_next = target(bns_t).gather(1, next_a)
                        y = br_t + cfg.gamma * (1.0 - bd_t) * q_next

                    loss = nn.functional.mse_loss(q_sa, y)

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(online.parameters(), 1.0)
                    opt.step()

                    soft_update(target, online, cfg.tau)

        eps = max(eps_min, eps * cfg.eps_decay)
        returns.append(ep_return)

    return online, returns


def eval_one_episode(env: BuildingEnergyEnv, model: QNet, device: torch.device, seed: int):
    s, _ = env.reset(seed=seed)
    done = False
    real_tap, algo_tap, temps = [], [], []
    ep_return = 0.0

    while not done:
        st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = model(st)
        a = int(torch.argmax(q, dim=1).item())

        ns, r, term, trunc, info = env.step(a)
        done = term or trunc

        ep_return += float(r)
        real_tap.append(info["real_tap"])
        algo_tap.append(info["dqn_tap"])
        temps.append(info["sim_temp"])
        s = ns

    return float(ep_return), np.array(real_tap), np.array(algo_tap), np.array(temps)


# =========================================================
# GRID SEARCH DQN
# =========================================================
def eval_config_dqn(df: pd.DataFrame, cfg: DQNConfig, device: torch.device):
    env_train = BuildingEnergyEnv(
        df,
        target_temp=TARGET_TEMP,
        comfort_range=COMFORT_RANGE,
        w_e=W_E,
        w_c=W_C,
        power_action_scale=POWER_ACTION_SCALE,
        seed=SEED,
        fixed_start_step=None,
    )

    model, returns = train_dqn_short(env_train, cfg, device, episodes=GRID_TRAIN_EPISODES)

    # Eval: múltiples episodios (ventanas) para distinguir configs
    env_eval = BuildingEnergyEnv(
        df,
        target_temp=TARGET_TEMP,
        comfort_range=COMFORT_RANGE,
        w_e=W_E,
        w_c=W_C,
        power_action_scale=POWER_ACTION_SCALE,
        seed=SEED,
        fixed_start_step=None,  # dejamos que varie por seed
    )

    all_costs = []
    all_energy_ratio = []
    all_viol_pct = []
    all_rmse = []
    all_kwh_real = []
    all_kwh_algo = []
    all_savings = []
    all_eval_return = []

    for i in range(EVAL_EPISODES):
        # si quieres que SIEMPRE evalúe la misma ventana base + offsets, usa FIXED_START_STEP:
        # env_eval.fixed_start_step = FIXED_START_STEP
        # y cambia seed igual para reset reproducible.
        seed_i = EVAL_SEED_BASE + i

        eval_return, real_tap, algo_tap, temp = eval_one_episode(env_eval, model, device, seed=seed_i)
        m = compute_metrics(env_eval, real_tap, algo_tap, temp)

        energy_ratio = m["kwh_algo"] / max(m["kwh_real"], 1e-9)
        viol_ratio = (m["viol_pct"] / 100.0)
        raw_cost = energy_ratio + (2.0 * viol_ratio)

        all_costs.append(raw_cost)
        all_energy_ratio.append(energy_ratio)
        all_viol_pct.append(m["viol_pct"])
        all_rmse.append(m["rmse_temp"])
        all_kwh_real.append(m["kwh_real"])
        all_kwh_algo.append(m["kwh_algo"])
        all_savings.append(m["savings_pct"])
        all_eval_return.append(eval_return)

    return {
        "energy_ratio": float(np.mean(all_energy_ratio)),
        "viol_pct": float(np.mean(all_viol_pct)),
        "rmse_temp": float(np.mean(all_rmse)),
        "raw_cost": float(np.mean(all_costs)),
        "kwh_real": float(np.mean(all_kwh_real)),
        "kwh_algo": float(np.mean(all_kwh_algo)),
        "savings_pct": float(np.mean(all_savings)),
        "reward_mean_last10": float(np.mean(returns[-10:])) if len(returns) >= 10 else float(np.mean(returns)),
        "eval_return": float(np.mean(all_eval_return)),
        "eval_cost_std": float(np.std(all_costs)),  # útil para ver estabilidad
    }


# =========================================================
# MAIN GRID
# =========================================================
if __name__ == "__main__":
    set_seed(SEED)

    device = torch.device("cpu") if FORCE_CPU else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device}")

    print(f"CSV: {CSV_PATH}")
    df = load_and_prepare(CSV_PATH)

    search_space = {
        "gamma":     [0.90, 0.95, 0.99],
        "lr":        [0.0001, 0.0003, 0.001],
        "eps_decay": [0.990, 0.995, 0.998],
        "tau":       [0.005, 0.01, 0.02],
    }

    fixed = {
        "batch_size": 64,
        "buffer_size": 50000,
        "start_learning": 200,
        "train_steps_per_ep": 1,
        "hidden": 256,
    }

    total = 1
    for k in search_space:
        total *= len(search_space[k])

    results = []
    count = 0
    start_time = time.time()

    for gamma in search_space["gamma"]:
        for lr in search_space["lr"]:
            for eps_decay in search_space["eps_decay"]:
                for tau in search_space["tau"]:
                    count += 1
                    print(f"\n[{count}/{total}] g={gamma}, lr={lr}, eps_decay={eps_decay}, tau={tau} ...", end="")

                    try:
                        cfg = DQNConfig(
                            gamma=gamma,
                            lr=lr,
                            eps_decay=eps_decay,
                            tau=tau,
                            batch_size=fixed["batch_size"],
                            buffer_size=fixed["buffer_size"],
                            start_learning=fixed["start_learning"],
                            train_steps_per_ep=fixed["train_steps_per_ep"],
                            hidden=fixed["hidden"],
                        )

                        m = eval_config_dqn(df, cfg, device)
                        results.append({
                            "gamma": gamma,
                            "lr": lr,
                            "eps_decay": eps_decay,
                            "tau": tau,
                            **fixed,
                            **m,
                        })

                        print(
                            f" energy_ratio={m['energy_ratio']:.3f}"
                            f" viol={m['viol_pct']:.2f}%"
                            f" rmse={m['rmse_temp']:.3f}"
                            f" cost={m['raw_cost']:.6f}"
                            f" (std={m['eval_cost_std']:.6f})",
                            end=""
                        )
                    except Exception as e:
                        print(f" ERROR: {e}")

    df_res = pd.DataFrame(results)
    if len(df_res) == 0:
        raise RuntimeError("No se generaron resultados. Revisa el CSV o el entorno.")

    # Normalización robusta
    cmin, cmax = df_res["raw_cost"].min(), df_res["raw_cost"].max()
    denom = (cmax - cmin)

    if denom < 1e-12:
        print("\n[WARN] raw_cost casi idéntico en todas las configs. score_0_1 por ranking.")
        df_res = df_res.sort_values("raw_cost", ascending=True).reset_index(drop=True)
        df_res["score_0_1"] = df_res.index / max(len(df_res) - 1, 1)
    else:
        df_res["score_0_1"] = (df_res["raw_cost"] - cmin) / denom

    df_res = df_res.sort_values("score_0_1", ascending=True)

    out_csv = f"grid_results_dqn_{os.path.splitext(NOMBRE_ARCHIVO)[0]}.csv"
    df_res.to_csv(os.path.join(OUT_DIR, out_csv), index=False)

    best = df_res.iloc[0]
    mins = (time.time() - start_time) / 60.0

    print("\n" + "=" * 70)
    print(f"TERMINADO en {mins:.1f} min | Mejor configuración (MENOR score):")
    print(best[[
        "gamma", "lr", "eps_decay", "tau",
        "energy_ratio", "viol_pct", "rmse_temp", "raw_cost", "score_0_1",
        "savings_pct", "eval_return", "eval_cost_std"
    ]])
    print(f"CSV guardado: {os.path.join(OUT_DIR, out_csv)}")
    print("=" * 70)

    # Gráfico Top 10
    top10 = df_res.head(10).copy()
    labels = [f"g={r.gamma},lr={r.lr},ed={r.eps_decay},tau={r.tau}" for r in top10.itertuples(index=False)]
    vals = top10["score_0_1"].values

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(vals)), vals)
    plt.ylim(0, 1.0)
    plt.title("Top 10 Configuraciones DQN (Score 0–1, menor es mejor)")
    plt.ylabel("Score (0..1)")
    plt.xlabel("Combinación de Hiperparámetros")
    plt.xticks(range(len(vals)), labels, rotation=30, ha="right")

    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    out_png = f"grid_top10_dqn_{os.path.splitext(NOMBRE_ARCHIVO)[0]}.png"
    plt.savefig(os.path.join(OUT_DIR, out_png), dpi=200)
    print(f"PNG guardado: {os.path.join(OUT_DIR, out_png)}")
    plt.show()

