from __future__ import annotations

import os
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt


# =========================================================
# Configuraciones
# =========================================================
NOMBRE_ARCHIVO = "viernes_hora.csv"
SEED = 42

# Inicio fijo del episodio (para evaluación y curva agente en robustez)
FIXED_START_STEP = 0  # 0 = primera ventana posible
FORCE_CPU = False

TARGET_TEMP = 23.0
COMFORT_RANGE = 1.5
W_E = 0.6
W_C = 0.4

# penalización por cambios bruscos de acción (suaviza control)
W_SMOOTH = 0.05  # 0.0 si no se desea suavizar

# SAC hiperparámetros
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
LR = 0.0003

EPISODES = 50
BATCH_SIZE = 256
BUFFER_SIZE = 100_000

START_STEPS = 0
UPDATE_AFTER = 0
HIDDEN = 256

ROBUSTNESS_EPISODES = 15

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, NOMBRE_ARCHIVO)
OUT_DIR = BASE_DIR
os.makedirs(OUT_DIR, exist_ok=True)

print(f"CSV: {CSV_PATH}")
print(f"Salida: {OUT_DIR}")


# =========================================================
# Reproducibilidad
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
# Carga de datos
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

    # (NUEVO) ordenar por Timestamp para robustez por días
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


# =========================================================
# Entorno Gymnasium
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
        w_smooth: float = 0.05,
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
        self.w_smooth = float(w_smooth)
        self.power_action_scale = float(power_action_scale)

        self.max_power = float(self.df["TAP"].max()) if float(self.df["TAP"].max()) > 0 else 1.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.action_space.seed(seed)

        self.start_step = 0
        self.current_step_index = 0

        # (NUEVO) para suavizar control
        self.prev_action = 0.0

        print(f" Entorno configurado para: {self.freq_name}. Pasos por episodio: {self.episode_length}")

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
        self.prev_action = 0.0
        return self._get_obs(self.start_step), {}

    def _get_obs(self, step_idx: int) -> np.ndarray:
        row = self.df.iloc[step_idx]
        return np.array(
            [row["TAP"], row["TF1"], row["TF2"], row["TF3"], row["HF1"], row["hour_sin"], row["hour_cos"]],
            dtype=np.float32,
        )

    def step(self, action):
        act_val = float(np.clip(action[0], -1.0, 1.0))
        real_idx = self.start_step + self.current_step_index

        next_row = self.df.iloc[real_idx + 1]
        base_tap = float(next_row["TAP"])
        base_temp_avg = float((next_row["TF1"] + next_row["TF2"] + next_row["TF3"]) / 3.0)

        power_mod = 1.0 + (act_val * self.power_action_scale)
        temp_mod = -(act_val * self.temp_inertia)

        sim_tap = max(0.0, base_tap * power_mod)
        sim_temp = base_temp_avg + temp_mod

        # =========================================================
        #  Recompensa con "zona muerta" en confort:
        # - Dentro del rango: penalización = 0
        # - Fuera del rango: penalización creciente (cuadrática suave)
        # + penalización por cambios bruscos de acción (suaviza)
        # =========================================================
        energy_saving = (base_tap - sim_tap) / self.max_power  # >0 ahorra, <0 consume más

        dist = abs(sim_temp - self.target_temp)

        # zona muerta: 0 dentro del confort
        if dist <= self.comfort_range:
            comfort_pen = 0.0
        else:
            # exceso sobre el rango (normalizado) y cuadrático
            excess = (dist - self.comfort_range) / max(self.comfort_range, 1e-9)
            comfort_pen = float(min(1.0, excess)) ** 2  # 0..1

        # suavidad (castigo cambios bruscos)
        smooth_pen = abs(act_val - float(self.prev_action))  # 0..2

        reward = (self.w_e * energy_saving) - (self.w_c * comfort_pen) - (self.w_smooth * smooth_pen)

        self.prev_action = act_val

        self.current_step_index += 1
        terminated = self.current_step_index >= self.episode_length

        next_obs = self._get_obs(real_idx + 1)
        next_obs[0] = sim_tap
        next_obs[1] = next_obs[1] + temp_mod
        next_obs[2] = next_obs[2] + temp_mod
        next_obs[3] = next_obs[3] + temp_mod

        info = {
            "real_tap": base_tap,
            "sac_tap": sim_tap,
            "sim_temp": sim_temp,
            "energy_saving": energy_saving,
            "comfort_pen": comfort_pen,
            "smooth_pen": smooth_pen,
        }
        return next_obs, float(reward), terminated, False, info


# =========================================================
# Replay Buffer
# =========================================================
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device, seed: int):
        self.cap = int(capacity)
        self.ptr = 0
        self.size = 0
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.s = np.zeros((self.cap, state_dim), dtype=np.float32)
        self.a = np.zeros((self.cap, action_dim), dtype=np.float32)
        self.r = np.zeros((self.cap, 1), dtype=np.float32)
        self.ns = np.zeros((self.cap, state_dim), dtype=np.float32)
        self.d = np.zeros((self.cap, 1), dtype=np.float32)

    def add(self, s, a, r, ns, d):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.ns[self.ptr] = ns
        self.d[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        s = torch.tensor(self.s[idx], device=self.device)
        a = torch.tensor(self.a[idx], device=self.device)
        r = torch.tensor(self.r[idx], device=self.device)
        ns = torch.tensor(self.ns[idx], device=self.device)
        d = torch.tensor(self.d[idx], device=self.device)
        return s, a, r, ns, d


# =========================================================
# SAC Networks
# =========================================================
LOG_STD_MIN = -20
LOG_STD_MAX = 2


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t

        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.02
    alpha: float = 0.05
    lr: float = 1e-4
    batch_size: int = 256
    buffer_size: int = 100_000
    start_steps: int = 0
    update_after: int = 0
    update_every: int = 1
    hidden: int = 256


class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: SACConfig, device: torch.device):
        self.device = device
        self.cfg = cfg

        self.q1 = MLP(state_dim + action_dim, 1, hidden=cfg.hidden).to(device)
        self.q2 = MLP(state_dim + action_dim, 1, hidden=cfg.hidden).to(device)
        self.q1_t = MLP(state_dim + action_dim, 1, hidden=cfg.hidden).to(device)
        self.q2_t = MLP(state_dim + action_dim, 1, hidden=cfg.hidden).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        self.pi = GaussianPolicy(state_dim, action_dim, hidden=cfg.hidden).to(device)

        self.q_opt = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr)
        self.pi_opt = optim.Adam(self.pi.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            mean, _ = self.pi(s)
            a = torch.tanh(mean)
        else:
            a, _, _ = self.pi.sample(s)
        return a.squeeze(0).cpu().numpy()

    def update(self, replay: ReplayBuffer):
        s, a, r, ns, d = replay.sample(self.cfg.batch_size)

        with torch.no_grad():
            na, logp_na, _ = self.pi.sample(ns)
            q1_t = self.q1_t(torch.cat([ns, na], dim=1))
            q2_t = self.q2_t(torch.cat([ns, na], dim=1))
            q_t = torch.min(q1_t, q2_t) - self.cfg.alpha * logp_na
            y = r + (1 - d) * self.cfg.gamma * q_t

        q1 = self.q1(torch.cat([s, a], dim=1))
        q2 = self.q2(torch.cat([s, a], dim=1))
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.q_opt.zero_grad()
        loss_q.backward()
        self.q_opt.step()

        na, logp_na, _ = self.pi.sample(s)
        q1_pi = self.q1(torch.cat([s, na], dim=1))
        q2_pi = self.q2(torch.cat([s, na], dim=1))
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (self.cfg.alpha * logp_na - q_pi).mean()

        self.pi_opt.zero_grad()
        loss_pi.backward()
        self.pi_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                pt.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                pt.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return float(loss_q.item()), float(loss_pi.item())


# =========================================================
# Entrenamiento / Evaluación
# =========================================================
def run_episode(env: BuildingEnergyEnv, agent: SACAgent, replay: Optional[ReplayBuffer], train: bool, total_steps: int):
    # Nota: env.reset usa fixed_start_step si está definido
    state, _ = env.reset(seed=SEED)
    done = False
    ep_reward = 0.0

    episode_info = {"real_tap": [], "sac_tap": [], "sim_temp": []}

    while not done:
        action = agent.act(state, deterministic=not train)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward

        episode_info["real_tap"].append(info["real_tap"])
        episode_info["sac_tap"].append(info["sac_tap"])
        episode_info["sim_temp"].append(info["sim_temp"])

        if train and replay is not None:
            replay.add(state, action, [reward], next_state, [float(done)])
            if replay.size >= agent.cfg.batch_size and total_steps >= agent.cfg.update_after:
                agent.update(replay)

        state = next_state
        total_steps += 1

    return ep_reward, total_steps, episode_info


def compute_metrics(env: BuildingEnergyEnv, ep_info: Dict[str, List[float]]):
    real = np.array(ep_info["real_tap"], dtype=float)
    sac = np.array(ep_info["sac_tap"], dtype=float)
    temp = np.array(ep_info["sim_temp"], dtype=float)

    kwh_real = real.sum() * (env.minutes_per_step / 60.0)
    kwh_sac = sac.sum() * (env.minutes_per_step / 60.0)
    savings_pct = ((kwh_real - kwh_sac) / max(kwh_real, 1e-6)) * 100.0  # >0 ahorra, <0 consume más

    low = env.target_temp - env.comfort_range
    high = env.target_temp + env.comfort_range
    viol = np.logical_or(temp < low, temp > high).astype(int)
    viol_pct = viol.mean() * 100.0

    rmse = math.sqrt(np.mean((temp - env.target_temp) ** 2))

    return {
        "kwh_real": float(kwh_real),
        "kwh_sac": float(kwh_sac),
        "savings_pct": float(savings_pct),
        "viol_pct": float(viol_pct),
        "rmse_temp": float(rmse),
    }


# =========================================================
# Plotting
# =========================================================
def savefig(path: str):
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_learning_curve(rewards: List[float], out_path: str):
    plt.figure()
    plt.title("Curva de Aprendizaje (Rewards) - SAC")
    plt.plot(rewards)
    plt.xlabel("Episodios")
    plt.ylabel("Return")
    savefig(out_path)


def plot_energy(env: BuildingEnergyEnv, real_tap: np.ndarray, sac_tap: np.ndarray, savings_pct: float, out_path: str):
    plt.figure()
    if savings_pct >= 0:
        titulo = f"Consumo Energético (Ahorro: {savings_pct:.1f}%) - SAC"
    else:
        titulo = f"Consumo Energético (Consumo: {abs(savings_pct):.1f}%) - SAC"
    plt.title(titulo)

    plt.plot(real_tap, linestyle="--", label="Línea Base (Histórico)")
    plt.plot(sac_tap, label="Agente SAC")
    plt.xlabel(f"Pasos de Tiempo ({env.freq_name})")
    plt.ylabel("Total Active Power (kW)")
    plt.legend()
    savefig(out_path)


def plot_comfort(env: BuildingEnergyEnv, temp: np.ndarray, out_path: str):
    low = env.target_temp - env.comfort_range
    high = env.target_temp + env.comfort_range

    plt.figure()
    plt.title("Control de Confort - SAC")
    plt.plot(temp, label="Temp Interior")
    plt.axhline(env.target_temp, linestyle="--", label=f"Target {env.target_temp:.0f}°C")
    plt.axhline(low, linestyle=":", label="Límite Inf")
    plt.axhline(high, linestyle=":", label="Límite Sup")
    plt.xlabel(f"Pasos de Tiempo ({env.freq_name})")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    savefig(out_path)


def plot_temp_rmse(env: BuildingEnergyEnv, temp: np.ndarray, rmse: float, out_path: str):
    low = env.target_temp - env.comfort_range
    high = env.target_temp + env.comfort_range

    plt.figure()
    plt.title(f"Desempeño del Control Térmico SAC\nRMSE: {rmse:.4f} °C")
    plt.plot(temp, label="Temperatura Interna")
    plt.axhline(env.target_temp, linestyle="--", label=f"Setpoint ({env.target_temp:.0f}°C)")
    plt.fill_between(np.arange(len(temp)), low, high, alpha=0.2, label=f"Zona de Confort (±{env.comfort_range:.1f}°C)")
    plt.xlabel(f"Time Steps (Intervalo: {env.freq_name})")
    plt.ylabel("Temperatura (°C)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    savefig(out_path)


# =========================================================
# (ARREGLADO) Robustez por días: curvas históricas alineadas
# =========================================================
def extract_daily_start_indices(df: pd.DataFrame, episode_length: int) -> List[int]:
    tmp = df.copy()
    tmp["date"] = tmp["Timestamp"].dt.date

    starts = []
    for _, g in tmp.groupby("date", sort=True):
        if len(g) >= episode_length:
            starts.append(int(g.index.min()))
    return starts


def plot_robustness(env: BuildingEnergyEnv, agent: SACAgent, df: pd.DataFrame, num_days: int, out_path: str):
    start_idxs = extract_daily_start_indices(df, env.episode_length)
    if not start_idxs:
        print("No hay días completos para robustez.")
        return

    # Selección reproducible y ordenada
    start_idxs = start_idxs[:min(len(start_idxs), num_days)]

    real_curves = []
    agent_curves = []

    for st in start_idxs:
        # Curva real: directamente del histórico (alineado por día)
        day_slice = env.df.iloc[st: st + env.episode_length]
        real_curves.append(day_slice["TAP"].to_numpy(dtype=float))

        # Curva agente: ejecutar episodio determinista en ese mismo día
        env.fixed_start_step = int(st)
        _, _, info_agent = run_episode(env, agent, replay=None, train=False, total_steps=10**9)
        agent_curves.append(np.array(info_agent["sac_tap"], dtype=float))

    real_curves = np.stack(real_curves, axis=0)        # [D, T]
    agent_curves = np.stack(agent_curves, axis=0)      # [D, T]
    agent_mean = agent_curves.mean(axis=0)

    plt.figure()
    plt.title(f"Robustez del Modelo (por días): {env.freq_name} - SAC")

    for i in range(real_curves.shape[0]):
        plt.plot(real_curves[i], alpha=0.25, linewidth=1.0,
                 label="Histórico (Variabilidad Diaria)" if i == 0 else None)

    plt.plot(agent_mean, linewidth=2.8, label="Agente SAC (Promedio)")
    plt.xlabel(f"Intervalos de Tiempo ({env.freq_name})")
    plt.ylabel("Consumo Energético (TAP) [kW]")
    plt.legend()
    savefig(out_path)

    # restaurar fijo para evaluación normal
    env.fixed_start_step = FIXED_START_STEP


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(SEED)

    if FORCE_CPU:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device}")

    df = load_and_prepare(CSV_PATH)

    env = BuildingEnergyEnv(
        df=df,
        target_temp=TARGET_TEMP,
        comfort_range=COMFORT_RANGE,
        w_e=W_E,
        w_c=W_C,
        w_smooth=W_SMOOTH,
        power_action_scale=0.6,
        seed=SEED,
        fixed_start_step=None,  # entrenamiento: random start (mejor generalización)
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    cfg = SACConfig(
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        lr=LR,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        start_steps=START_STEPS,
        update_after=UPDATE_AFTER,
        update_every=1,
        hidden=HIDDEN,
    )

    agent = SACAgent(state_dim, action_dim, cfg, device)
    replay = ReplayBuffer(cfg.buffer_size, state_dim, action_dim, device=device, seed=SEED)

    rewards = []
    total_steps = 0
    t0 = time.time()

    # Entrenamiento
    for ep in range(EPISODES):
        # para entrenamiento, start aleatorio (env.fixed_start_step=None)
        env.fixed_start_step = None
        ep_reward, total_steps, _ = run_episode(env, agent, replay=replay, train=True, total_steps=total_steps)
        rewards.append(ep_reward)

        if (ep + 1) % max(1, EPISODES // 10) == 0:
            print(f"Ep {ep+1}/{EPISODES} | Return: {ep_reward:.3f}")

    print(f"Entrenamiento SAC completado en {(time.time() - t0):.1f}s")

    # Evaluación fija (misma ventana siempre)
    env.fixed_start_step = FIXED_START_STEP
    _, _, ep_info = run_episode(env, agent, replay=None, train=False, total_steps=10**9)
    metrics = compute_metrics(env, ep_info)

    print("\nRESULTADO FINAL (SAC):")
    print(f" - Frecuencia: {env.freq_name} ({env.minutes_per_step} min/step)")
    print(f" - Energía real (kWh): {metrics['kwh_real']:.2f}")
    print(f" - Energía SAC  (kWh): {metrics['kwh_sac']:.2f}")

    # (ARREGLADO) Ahorro vs Consumo
    if metrics["savings_pct"] >= 0:
        print(f" - Ahorro vs histórico (%): {metrics['savings_pct']:.2f}")
    else:
        print(f" - Consumo extra vs histórico (%): {abs(metrics['savings_pct']):.2f}")

    print(f" - Violaciones confort (%): {metrics['viol_pct']:.2f}")
    print(f" - RMSE temperatura (°C): {metrics['rmse_temp']:.4f}")

    # Guardar CSV resultados
    metrics_path = os.path.join(OUT_DIR, "resultado_sac.csv")
    pd.DataFrame([{
        "csv_path": CSV_PATH,
        "freq_name": env.freq_name,
        "minutes_per_step": env.minutes_per_step,
        "seed": SEED,
        "fixed_start_step": FIXED_START_STEP,
        "episodes": EPISODES,
        "gamma": cfg.gamma,
        "tau": cfg.tau,
        "alpha": cfg.alpha,
        "lr": cfg.lr,
        "w_e": W_E,
        "w_c": W_C,
        "w_smooth": W_SMOOTH,
        **metrics,
    }]).to_csv(metrics_path, index=False)
    print(f"CSV guardado: {metrics_path}")

    # Plots
    plot_learning_curve(rewards, os.path.join(OUT_DIR, "curva_aprendizaje_rewards_sac.png"))

    real_tap = np.array(ep_info["real_tap"], dtype=float)
    sac_tap = np.array(ep_info["sac_tap"], dtype=float)
    temp = np.array(ep_info["sim_temp"], dtype=float)

    plot_energy(env, real_tap, sac_tap, metrics["savings_pct"], os.path.join(OUT_DIR, "consumo_energetico_sac.png"))
    plot_comfort(env, temp, os.path.join(OUT_DIR, "control_confort_sac.png"))
    plot_temp_rmse(env, temp, metrics["rmse_temp"], os.path.join(OUT_DIR, "desempeno_control_termico_rmse_sac.png"))

    # Robustez por días (ARREGLADO)
    plot_robustness(
        env=env,
        agent=agent,
        df=df,
        num_days=ROBUSTNESS_EPISODES,
        out_path=os.path.join(OUT_DIR, "robustez_modelo_sac.png")
    )

    print(f"Gráficos guardados en: {OUT_DIR}")


if __name__ == "__main__":
    main()

