from __future__ import annotations

import os
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt



# CONFIGURACIÓN GENERAL
NOMBRE_ARCHIVO = "miercoles.csv"
SEED = 42

TARGET_TEMP = 23.0
COMFORT_RANGE = 1.5

W_E = 0.6
W_C = 0.4

POWER_ACTION_SCALE = 0.6

# 4 HIPERPARÁMETROS PRINCIPALES 
GAMMA = 0.95
LR = 0.0001
EPS_DECAY = 0.998   # ed 
TAU = 0.02          # soft update del target

# Secundarios 
BATCH_SIZE = 256
BUFFER_SIZE = 100_000
EPISODES = 120

# Con TAU ya no es obligatorio usar hard update
TARGET_UPDATE_EVERY = 500

EPS_START = 1.0
EPS_END = 0.05

START_STEPS = 1_000
TRAIN_AFTER = 1_000
TRAIN_EVERY = 1

ROBUSTNESS_DAYS = 15

ACTION_VALUES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)



# RUTAS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, NOMBRE_ARCHIVO)
OUT_DIR = BASE_DIR
os.makedirs(OUT_DIR, exist_ok=True)

print(f" CSV: {CSV_PATH}")
print(f" Salida: {OUT_DIR}")



# REPRODUCIBILIDAD (RESULTADOS FIJOS)
def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
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



# PREPROCESADO
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

    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df



# ENTORNO (DQN: acciones discretas)
class BuildingEnergyEnvDQN(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        action_values: np.ndarray,
        target_temp: float = 23.0,
        comfort_range: float = 1.5,
        w_e: float = 0.6,
        w_c: float = 0.4,
        power_action_scale: float = 0.6,
        seed: int = 42,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)

        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)

        self.action_values = np.array(action_values, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_values))
        self.action_space.seed(self.base_seed)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

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

        self.start_step = 0
        self.current_step_index = 0

        print(f"Entorno DQN configurado para: {self.freq_name}. Pasos por episodio: {self.episode_length}")
        print(f"Acciones discretas: {self.action_values.tolist()}")

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            seed = int(seed)
            self.rng = np.random.default_rng(seed)
            self.action_space.seed(seed)

        max_idx = len(self.df) - self.episode_length - 2
        if max_idx <= 0:
            raise ValueError("Dataset no tiene suficientes filas para un episodio completo.")

        self.start_step = int(self.rng.integers(0, max_idx))
        self.current_step_index = 0
        return self._get_obs(self.start_step), {}

    def reset_at(self, start_step: int):
        max_idx = len(self.df) - self.episode_length - 2
        start_step = int(max(0, min(start_step, max_idx)))
        self.start_step = start_step
        self.current_step_index = 0
        return self._get_obs(self.start_step)

    def _get_obs(self, step_idx: int) -> np.ndarray:
        row = self.df.iloc[step_idx]
        return np.array(
            [row["TAP"], row["TF1"], row["TF2"], row["TF3"], row["HF1"], row["hour_sin"], row["hour_cos"]],
            dtype=np.float32,
        )

    def step(self, action: int):
        action = int(action)
        act_val = float(self.action_values[action])
        real_idx = self.start_step + self.current_step_index

        next_row = self.df.iloc[real_idx + 1]
        base_tap = float(next_row["TAP"])
        base_temp_avg = float((next_row["TF1"] + next_row["TF2"] + next_row["TF3"]) / 3.0)

        power_mod = 1.0 + (act_val * self.power_action_scale)
        temp_mod = -(act_val * self.temp_inertia)

        sim_tap = max(0.0, base_tap * power_mod)
        sim_temp = base_temp_avg + temp_mod

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



# REPLAY BUFFER
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.cap = int(capacity)
        self.device = device
        self.ptr = 0
        self.size = 0

        self.s = np.zeros((self.cap, state_dim), dtype=np.float32)
        self.a = np.zeros((self.cap, 1), dtype=np.int64)
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
        idx = np.random.randint(0, self.size, size=batch_size)
        s = torch.tensor(self.s[idx], device=self.device)
        a = torch.tensor(self.a[idx], device=self.device)
        r = torch.tensor(self.r[idx], device=self.device)
        ns = torch.tensor(self.ns[idx], device=self.device)
        d = torch.tensor(self.d[idx], device=self.device)
        return s, a, r, ns, d



# DQN NETWORK
class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# soft update (usa TAU del grid)
def soft_update(target_net: nn.Module, policy_net: nn.Module, tau: float) -> None:
    for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
        tp.data.copy_(tau * pp.data + (1.0 - tau) * tp.data)



# MÉTRICAS Y GRÁFICOS
def compute_metrics(env: BuildingEnergyEnvDQN, ep_info: Dict[str, List[float]]):
    real = np.array(ep_info["real_tap"], dtype=float)
    dqn = np.array(ep_info["dqn_tap"], dtype=float)
    temp = np.array(ep_info["sim_temp"], dtype=float)

    kwh_real = real.sum() * (env.minutes_per_step / 60.0)
    kwh_dqn = dqn.sum() * (env.minutes_per_step / 60.0)
    savings_pct = ((kwh_real - kwh_dqn) / max(kwh_real, 1e-6)) * 100.0

    low = env.target_temp - env.comfort_range
    high = env.target_temp + env.comfort_range
    viol = np.logical_or(temp < low, temp > high).astype(int)
    viol_pct = viol.mean() * 100.0

    rmse = math.sqrt(np.mean((temp - env.target_temp) ** 2))

    return {
        "kwh_real": float(kwh_real),
        "kwh_dqn": float(kwh_dqn),
        "savings_pct": float(savings_pct),
        "viol_pct": float(viol_pct),
        "rmse_temp": float(rmse),
    }


def savefig(path: str):
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_learning_curve(rewards: List[float], out_path: str):
    plt.figure()
    plt.title("Curva de Aprendizaje (Rewards) - DQN")
    plt.plot(rewards)
    plt.xlabel("Episodios")
    plt.ylabel("Return")
    savefig(out_path)


def plot_energy(env: BuildingEnergyEnvDQN, real_tap: np.ndarray, dqn_tap: np.ndarray, savings_pct: float, out_path: str):
    if savings_pct >= 0:
        title = f"Consumo Energético (Ahorro: {savings_pct:.1f}%) - DQN"
    else:
        title = f"Consumo Energético (Aumento: {abs(savings_pct):.1f}%) - DQN"

    plt.figure()
    plt.title(title)
    plt.plot(real_tap, linestyle="--", label="Línea Base (Histórico)")
    plt.plot(dqn_tap, label="Agente DQN")
    plt.xlabel(f"Pasos de Tiempo ({env.freq_name})")
    plt.ylabel("Total Active Power (kW)")
    plt.legend()
    savefig(out_path)


def plot_temp_rmse(env: BuildingEnergyEnvDQN, temp: np.ndarray, rmse: float, out_path: str):
    low = env.target_temp - env.comfort_range
    high = env.target_temp + env.comfort_range

    plt.figure()
    plt.title(f"Desempeño del Control Térmico DQN\nRMSE: {rmse:.4f} °C")
    plt.plot(temp, label="Temperatura Interna")
    plt.axhline(env.target_temp, linestyle="--", label=f"Setpoint ({env.target_temp:.0f}°C)")
    plt.fill_between(np.arange(len(temp)), low, high, alpha=0.2, label=f"Zona de Confort (±{env.comfort_range:.1f}°C)")
    plt.xlabel(f"Time Steps (Intervalo: {env.freq_name})")
    plt.ylabel("Temperatura (°C)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    savefig(out_path)


def plot_comfort(env: BuildingEnergyEnvDQN, temp: np.ndarray, out_path: str):
    low = env.target_temp - env.comfort_range
    high = env.target_temp + env.comfort_range

    plt.figure()
    plt.title("Control de Confort - DQN")
    plt.plot(temp, label="Temp Interior")
    plt.axhline(env.target_temp, linestyle="--", label=f"Target {env.target_temp:.0f}°C")
    plt.axhline(low, linestyle=":", label="Límite Inf")
    plt.axhline(high, linestyle=":", label="Límite Sup")
    plt.xlabel(f"Pasos de Tiempo ({env.freq_name})")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    savefig(out_path)


# EPISODIO (permite start_step fijo o seed fijo)
def run_episode(env: BuildingEnergyEnvDQN, policy_net: DQN, device: torch.device,
                epsilon: float, train: bool, seed: Optional[int] = None, start_step: Optional[int] = None):
    if start_step is not None:
        state = env.reset_at(start_step)
    else:
        state, _ = env.reset(seed=seed)

    done = False
    ep_reward = 0.0
    ep_info = {"real_tap": [], "dqn_tap": [], "sim_temp": []}

    while not done:
        if train and random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q = policy_net(s)
                action = int(torch.argmax(q, dim=1).item())

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward

        ep_info["real_tap"].append(info["real_tap"])
        ep_info["dqn_tap"].append(info["dqn_tap"])
        ep_info["sim_temp"].append(info["sim_temp"])

        state = next_state

    return ep_reward, ep_info


# ROBUSTEZ POR DÍAS
def extract_daily_start_indices(df: pd.DataFrame, episode_length: int) -> List[int]:
    tmp = df.copy()
    tmp["date"] = tmp["Timestamp"].dt.date

    starts = []
    for _, g in tmp.groupby("date", sort=True):
        if len(g) >= episode_length:
            starts.append(int(g.index.min()))
    return starts


def plot_robustness(env: BuildingEnergyEnvDQN, policy_net: DQN, device: torch.device, df: pd.DataFrame, out_path: str):
    start_idxs = extract_daily_start_indices(df, env.episode_length)
    if not start_idxs:
        print("No hay días completos para robustez.")
        return

    start_idxs = start_idxs[:min(len(start_idxs), ROBUSTNESS_DAYS)]

    real_curves = []
    agent_curves = []

    for st in start_idxs:
        day_slice = env.df.iloc[st: st + env.episode_length]
        real_curves.append(day_slice["TAP"].to_numpy(dtype=float))

        _, info_agent = run_episode(env, policy_net, device, epsilon=0.0, train=False, start_step=st)
        agent_curves.append(np.array(info_agent["dqn_tap"], dtype=float))

    real_curves = np.stack(real_curves, axis=0)
    agent_curves = np.stack(agent_curves, axis=0)
    agent_mean = agent_curves.mean(axis=0)

    plt.figure()
    plt.title(f"Robustez del Modelo: {env.freq_name} - DQN")

    for i in range(real_curves.shape[0]):
        plt.plot(real_curves[i], alpha=0.25, linewidth=1.0,
                 label="Histórico (Variabilidad Diaria)" if i == 0 else None)

    plt.plot(agent_mean, linewidth=2.8, label="Agente DQN (Promedio)")
    plt.xlabel(f"Intervalos de Tiempo ({env.freq_name})")
    plt.ylabel("Consumo Energético (TAP) [kW]")
    plt.legend()
    savefig(out_path)


# MAIN
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device}")

    df = load_and_prepare(CSV_PATH)

    env = BuildingEnergyEnvDQN(
        df=df,
        action_values=ACTION_VALUES,
        target_temp=TARGET_TEMP,
        comfort_range=COMFORT_RANGE,
        w_e=W_E,
        w_c=W_C,
        power_action_scale=POWER_ACTION_SCALE,
        seed=SEED,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # LR del grid
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer(BUFFER_SIZE, state_dim, device=device)

    rewards = []
    total_steps = 0
    t0 = time.time()

    # epsilon inicial
    epsilon = EPS_START

    for ep in range(EPISODES):
        state, _ = env.reset(seed=SEED + ep)
        done = False
        ep_reward = 0.0

        while not done:
            # exploración inicial
            if total_steps < START_STEPS:
                action = env.action_space.sample()
            else:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        q = policy_net(s)
                        action = int(torch.argmax(q, dim=1).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            replay.add(state, [action], [reward], next_state, [float(done)])

            state = next_state
            total_steps += 1

            # entrenamiento
            if replay.size >= BATCH_SIZE and total_steps >= TRAIN_AFTER and (total_steps % TRAIN_EVERY == 0):
                s, a, r, ns, d = replay.sample(BATCH_SIZE)

                q_sa = policy_net(s).gather(1, a)

                with torch.no_grad():
                    max_next_q = target_net(ns).max(dim=1, keepdim=True)[0]
                    # GAMMA del grid
                    target = r + (1 - d) * GAMMA * max_next_q

                loss = F.mse_loss(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # TAU del grid (soft update)
                soft_update(target_net, policy_net, TAU)

            #hard update adicional (no es necesario con TAU)
            #if TARGET_UPDATE_EVERY and (total_steps % TARGET_UPDATE_EVERY == 0):
                # Puedes comentarlo si quieres SOLO TAU
                #target_net.load_state_dict(policy_net.state_dict())

        rewards.append(ep_reward)

        # EPS_DECAY del grid (por episodio)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if (ep + 1) % max(1, EPISODES // 10) == 0:
            print(f"Ep {ep+1}/{EPISODES} | Return: {ep_reward:.3f} | Epsilon: {epsilon:.3f}")

    print(f"Entrenamiento DQN completado en {(time.time() - t0):.1f}s")

    # evaluación
    _, ep_info = run_episode(env, policy_net, device, epsilon=0.0, train=False, seed=SEED)
    metrics = compute_metrics(env, ep_info)

    print("\nRESULTADO FINAL (DQN):")
    print(f" - Frecuencia: {env.freq_name} ({env.minutes_per_step} min/step)")
    print(f" - Energía real (kWh): {metrics['kwh_real']:.2f}")
    print(f" - Energía DQN  (kWh): {metrics['kwh_dqn']:.2f}")
    if metrics["savings_pct"] >= 0:
        print(f" - Ahorro vs histórico (%): {metrics['savings_pct']:.2f}")
    else:
        print(f" - Aumento de consumo (%): {abs(metrics['savings_pct']):.2f}")
    print(f" - Violaciones confort (%): {metrics['viol_pct']:.2f}")
    print(f" - RMSE temperatura (°C): {metrics['rmse_temp']:.4f}")

    metrics_path = os.path.join(OUT_DIR, "resultado_dqn.csv")
    pd.DataFrame([{
        "csv_path": CSV_PATH,
        "freq_name": env.freq_name,
        "minutes_per_step": env.minutes_per_step,

        # 4 hiperparámetros principales guardados
        "gamma": GAMMA,
        "lr": LR,
        "eps_decay": EPS_DECAY,
        "tau": TAU,

        # secundarios 
        "batch_size": BATCH_SIZE,
        "buffer_size": BUFFER_SIZE,
        "episodes": EPISODES,
        "actions": str(ACTION_VALUES.tolist()),

        **metrics,
    }]).to_csv(metrics_path, index=False)
    print(f"CSV guardado: {metrics_path}")

    # plots
    plot_learning_curve(rewards, os.path.join(OUT_DIR, "curva_aprendizaje_rewards_dqn.png"))

    real_tap = np.array(ep_info["real_tap"], dtype=float)
    dqn_tap = np.array(ep_info["dqn_tap"], dtype=float)
    temp = np.array(ep_info["sim_temp"], dtype=float)

    plot_energy(env, real_tap, dqn_tap, metrics["savings_pct"], os.path.join(OUT_DIR, "consumo_energetico_dqn.png"))
    plot_comfort(env, temp, os.path.join(OUT_DIR, "control_confort_dqn.png"))
    plot_temp_rmse(env, temp, metrics["rmse_temp"], os.path.join(OUT_DIR, "desempeno_control_termico_rmse_dqn.png"))

    plot_robustness(env, policy_net, device, df=df, out_path=os.path.join(OUT_DIR, "robustez_modelo_dqn.png"))

    print(f"Gráficos guardados en: {OUT_DIR}")


if __name__ == "__main__":
    main()

