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


# CONFIG
NOMBRE_ARCHIVO = "viernes_hora.csv"
SEED = 42

# Para resultados "fijos" (en evaluación y robustez)
FIXED_START_STEP = 0   # 0 = siempre desde la primera ventana posible
FORCE_CPU = False      # True = más estable (pero más lento)

# Entorno
TARGET_TEMP = 23.0
COMFORT_RANGE = 1.5
W_E = 0.6
W_C = 0.4
POWER_ACTION_SCALE = 0.6

# PPO hiperparámetros
EPISODES = 200
GAMMA = 0.99
GAE_LAMBDA = 0.9
LR = 0.0001
CLIP_EPS = 0.1
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 10
MINIBATCH_SIZE = 256
HIDDEN = 256

# Robustez
ROBUSTNESS_EPISODES = 15

#  Reward/Return métrica
RETURN_LAST_N = 20              # promedio y std de últimos N episodios
EVAL_REPEAT = 3                 # repite evaluación fija N veces

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, NOMBRE_ARCHIVO)
OUT_DIR = BASE_DIR
os.makedirs(OUT_DIR, exist_ok=True)

print(f"CSV: {CSV_PATH}")
print(f"Salida: {OUT_DIR}")



# Reproducibilidad
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



# Carga de datos
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



# Entorno
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

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        self.action_space.seed(seed)

        self.start_step = 0
        self.current_step_index = 0

        print(f"Entorno configurado para: {self.freq_name}. Pasos por episodio: {self.episode_length}")

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

        info = {
            "real_tap": base_tap,
            "ppo_tap": sim_tap,
            "sim_temp": sim_temp,
            "energy_saving": energy_saving,
            "viol_norm": viol_norm,
        }
        return next_obs, float(reward), terminated, False, info



# PPO Model
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        mu = self.mu(h)
        v = self.v(h)
        log_std = self.log_std.expand_as(mu)
        return mu, log_std, v

    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None):
        mu, log_std, v = self.forward(x)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)

        if action is None:
            z = dist.rsample()
            a = torch.tanh(z)
        else:
            a = action
            eps = 1e-6
            z = 0.5 * torch.log((1 + torch.clamp(a, -1 + eps, 1 - eps)) / (1 - torch.clamp(a, -1 + eps, 1 - eps)))

        log_prob = dist.log_prob(z) - torch.log(1 - torch.tanh(z).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return a, log_prob, entropy, v.squeeze(-1)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 256
    hidden: int = 256


# Rollout buffer
class RolloutBuffer:
    def __init__(self, size: int, state_dim: int, action_dim: int, device: torch.device):
        self.size = size
        self.device = device
        self.ptr = 0

        self.s = torch.zeros((size, state_dim), device=device)
        self.a = torch.zeros((size, action_dim), device=device)
        self.logp = torch.zeros((size,), device=device)
        self.r = torch.zeros((size,), device=device)
        self.d = torch.zeros((size,), device=device)
        self.v = torch.zeros((size,), device=device)

        self.adv = torch.zeros((size,), device=device)
        self.ret = torch.zeros((size,), device=device)

    def add(self, s, a, logp, r, d, v):
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.logp[i] = logp
        self.r[i] = r
        self.d[i] = d
        self.v[i] = v
        self.ptr += 1

    def compute_gae(self, last_value: torch.Tensor, cfg: PPOConfig):
        gae = 0.0
        for t in reversed(range(self.size)):
            next_nonterminal = 1.0 - self.d[t]
            next_value = last_value if t == self.size - 1 else self.v[t + 1]
            delta = self.r[t] + cfg.gamma * next_value * next_nonterminal - self.v[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * gae
            self.adv[t] = gae
        self.ret = self.adv + self.v
        self.adv = (self.adv - self.adv.mean()) / (self.adv.std() + 1e-8)

    def get_minibatches(self, batch_size: int):
        idx = torch.randperm(self.size, device=self.device)
        for start in range(0, self.size, batch_size):
            mb = idx[start:start + batch_size]
            yield self.s[mb], self.a[mb], self.logp[mb], self.adv[mb], self.ret[mb], self.v[mb]


# Métricas & plots
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


def savefig(path: str):
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_learning_curve(rewards: List[float], out_path: str):
    plt.figure()
    plt.title("Curva de Aprendizaje (Rewards) - PPO")
    plt.plot(rewards)
    plt.xlabel("Episodios")
    plt.ylabel("Return")
    savefig(out_path)


def plot_energy(env, real_tap, algo_tap, savings_pct, out_path, algo_name="PPO"):
    plt.figure()
    if savings_pct >= 0:
        titulo = f"Consumo Energético (Ahorro: {savings_pct:.1f}%) - {algo_name}"
    else:
        titulo = f"Consumo Energético (Consumo: {abs(savings_pct):.1f}%) - {algo_name}"
    plt.title(titulo)

    plt.plot(real_tap, linestyle="--", label="Línea Base (Histórico)")
    plt.plot(algo_tap, label=f"Agente {algo_name}")
    plt.xlabel(f"Pasos de Tiempo ({env.freq_name})")
    plt.ylabel("Total Active Power (kW)")
    plt.legend()
    savefig(out_path)


def plot_comfort(env: BuildingEnergyEnv, temp: np.ndarray, out_path: str):
    low = env.target_temp - env.comfort_range
    high = env.target_temp + env.comfort_range

    plt.figure()
    plt.title("Control de Confort - PPO")
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
    plt.title(f"Desempeño del Control Térmico PPO\nRMSE: {rmse:.4f} °C")
    plt.plot(temp, label="Temperatura Interna")
    plt.axhline(env.target_temp, linestyle="--", label=f"Setpoint ({env.target_temp:.0f}°C)")
    plt.fill_between(np.arange(len(temp)), low, high, alpha=0.2, label=f"Zona de Confort (±{env.comfort_range:.1f}°C)")
    plt.xlabel(f"Time Steps (Intervalo: {env.freq_name})")
    plt.ylabel("Temperatura (°C)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    savefig(out_path)


def plot_robustness(env: BuildingEnergyEnv, model: ActorCritic, device: torch.device, num_episodes: int, out_path: str):
    rng = np.random.default_rng(SEED)
    curves = []

    for _ in range(num_episodes):
        env.fixed_start_step = int(rng.integers(0, max(1, len(env.df) - env.episode_length - 2)))
        s, _ = env.reset(seed=SEED)
        done = False
        real_curve = []
        while not done:
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a, _, _, _ = model.get_action_and_value(st)
            ns, _, term, trunc, info = env.step(a.squeeze(0).cpu().numpy())
            done = term or trunc
            real_curve.append(info["real_tap"])
            s = ns
        curves.append(np.array(real_curve, dtype=float))

    curves = np.stack(curves, axis=0)

    env.fixed_start_step = FIXED_START_STEP
    s, _ = env.reset(seed=SEED)
    done = False
    algo_curve = []
    while not done:
        st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a, _, _, _ = model.get_action_and_value(st)
        ns, _, term, trunc, info = env.step(a.squeeze(0).cpu().numpy())
        done = term or trunc
        algo_curve.append(info["ppo_tap"])
        s = ns
    algo_curve = np.array(algo_curve, dtype=float)

    plt.figure()
    plt.title(f"Robustez del Modelo: {env.freq_name} - PPO")
    for i in range(curves.shape[0]):
        plt.plot(curves[i], alpha=0.25, linewidth=1.0, label="Histórico (Variabilidad Diaria)" if i == 0 else None)
    plt.plot(algo_curve, linewidth=2.5, label="Agente PPO (Optimizado)")
    plt.xlabel(f"Intervalos de Tiempo ({env.freq_name})")
    plt.ylabel("Consumo Energético (TAP) [kW]")
    plt.legend()
    savefig(out_path)



#Reward/Return:(entrenamiento + evaluación)
def summarize_training_returns(returns: List[float], last_n: int = 20) -> Dict[str, float]:
    arr = np.array(returns, dtype=float)
    last_n = int(min(max(1, last_n), len(arr)))
    out = {
        "train_return_avg_lastN": float(arr[-last_n:].mean()),
        "train_return_std_lastN": float(arr[-last_n:].std()),
        "train_return_best": float(arr.max()),
        "train_return_last": float(arr[-1]),
        "train_return_lastN": float(last_n),
    }
    return out


def eval_one_episode_return(env: BuildingEnergyEnv, model: ActorCritic, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evalúa 1 episodio con FIXED_START_STEP y devuelve:
    - eval_return (suma de rewards del episodio)
    - real_tap, algo_tap, temp (para métricas físicas)
    """
    env.fixed_start_step = FIXED_START_STEP
    s, _ = env.reset(seed=SEED)

    done = False
    real_tap, algo_tap, temps = [], [], []
    eval_return = 0.0

    while not done:
        st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a, _, _, _ = model.get_action_and_value(st)
        ns, r, term, trunc, info = env.step(a.squeeze(0).cpu().numpy())
        done = term or trunc

        eval_return += float(r)
        real_tap.append(info["real_tap"])
        algo_tap.append(info["ppo_tap"])
        temps.append(info["sim_temp"])
        s = ns

    return eval_return, np.array(real_tap, dtype=float), np.array(algo_tap, dtype=float), np.array(temps, dtype=float)


def eval_return_repeated(env: BuildingEnergyEnv, model: ActorCritic, device: torch.device, repeats: int = 3) -> Dict[str, float]:
    """
    Repite la evaluación fija varias veces (misma ventana) para reportar promedio y std.
    OJO: misma ventana => si todo es determinista, el std puede salir ~0, pero sirve como métrica formal.
    """
    vals = []
    last_pack = None
    for _ in range(max(1, int(repeats))):
        pack = eval_one_episode_return(env, model, device)
        vals.append(pack[0])
        last_pack = pack

    vals = np.array(vals, dtype=float)
    return {
        "eval_return_mean": float(vals.mean()),
        "eval_return_std": float(vals.std()),
        "eval_return_repeats": float(len(vals)),
    }



# PPO Train / Eval
def train_ppo(env: BuildingEnergyEnv, cfg: PPOConfig, device: torch.device):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCritic(state_dim, action_dim, hidden=cfg.hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    episode_returns: List[float] = []

    t0 = time.time()

    for ep in range(EPISODES):
        s, _ = env.reset(seed=SEED)
        buf = RolloutBuffer(env.episode_length, state_dim, action_dim, device)

        ep_return = 0.0

        for _ in range(env.episode_length):
            st = torch.tensor(s, dtype=torch.float32, device=device)
            with torch.no_grad():
                a, logp, ent, v = model.get_action_and_value(st.unsqueeze(0))
            a_np = a.squeeze(0).cpu().numpy()

            ns, r, term, trunc, _info = env.step(a_np)
            done = term or trunc

            buf.add(
                s=st,
                a=torch.tensor(a_np, dtype=torch.float32, device=device),
                logp=logp.squeeze(0),
                r=torch.tensor(r, dtype=torch.float32, device=device),
                d=torch.tensor(float(done), dtype=torch.float32, device=device),
                v=v.squeeze(0),
            )

            ep_return += float(r)
            s = ns

            if done:
                break

        with torch.no_grad():
            st_last = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, _, v_last = model.get_action_and_value(st_last)
        buf.compute_gae(last_value=v_last.squeeze(0), cfg=cfg)

        for _ in range(cfg.update_epochs):
            for mb_s, mb_a, mb_logp_old, mb_adv, mb_ret, _mb_v_old in buf.get_minibatches(cfg.minibatch_size):
                _new_a, new_logp, entropy, new_v = model.get_action_and_value(mb_s, action=mb_a)

                ratio = torch.exp(new_logp - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                loss_pi = -torch.min(surr1, surr2).mean()

                loss_v = F.mse_loss(new_v, mb_ret)
                loss_ent = -entropy.mean()

                loss = loss_pi + cfg.vf_coef * loss_v + cfg.ent_coef * loss_ent

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

        episode_returns.append(ep_return)

        if (ep + 1) % max(1, EPISODES // 10) == 0:
            print(f"Ep {ep+1}/{EPISODES} | Return: {ep_return:.3f}")

    print(f"PPO entrenamiento completado en {(time.time()-t0):.1f}s")
    return model, episode_returns


def eval_one_episode(env: BuildingEnergyEnv, model: ActorCritic, device: torch.device):
    env.fixed_start_step = FIXED_START_STEP
    s, _ = env.reset(seed=SEED)

    done = False
    real_tap, algo_tap, temps = [], [], []

    while not done:
        st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a, _, _, _ = model.get_action_and_value(st)
        ns, _, term, trunc, info = env.step(a.squeeze(0).cpu().numpy())
        done = term or trunc
        real_tap.append(info["real_tap"])
        algo_tap.append(info["ppo_tap"])
        temps.append(info["sim_temp"])
        s = ns

    return np.array(real_tap, dtype=float), np.array(algo_tap, dtype=float), np.array(temps, dtype=float)



# Main
def main():
    set_seed(SEED)

    if FORCE_CPU:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device}")

    df = load_and_prepare(CSV_PATH)

    env = BuildingEnergyEnv(
        df,
        target_temp=TARGET_TEMP,
        comfort_range=COMFORT_RANGE,
        w_e=W_E,
        w_c=W_C,
        power_action_scale=POWER_ACTION_SCALE,
        seed=SEED,
        fixed_start_step=FIXED_START_STEP,
    )

    cfg = PPOConfig(
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        lr=LR,
        clip_eps=CLIP_EPS,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        update_epochs=UPDATE_EPOCHS,
        minibatch_size=MINIBATCH_SIZE,
        hidden=HIDDEN,
    )

    model, returns = train_ppo(env, cfg, device)

    # Reward/Return (entrenamiento) 
    train_reward_stats = summarize_training_returns(returns, last_n=RETURN_LAST_N)

    #  Evaluación fija + reward/return evaluado
    eval_reward_stats = eval_return_repeated(env, model, device, repeats=EVAL_REPEAT)

    # Métricas físicas (energía, confort, RMSE) usando 1 evaluación (la última)
    eval_return, real_tap, algo_tap, temp = eval_one_episode_return(env, model, device)
    metrics = compute_metrics(env, real_tap, algo_tap, temp)

    print("\nRESULTADO FINAL (PPO):")
    print(f" - Frecuencia: {env.freq_name} ({env.minutes_per_step} min/step)")
    print(f" - Energía real (kWh): {metrics['kwh_real']:.2f}")
    print(f" - Energía PPO  (kWh): {metrics['kwh_algo']:.2f}")
    print(f" - Ahorro vs histórico (%): {metrics['savings_pct']:.2f}")
    print(f" - Violaciones confort (%): {metrics['viol_pct']:.2f}")
    print(f" - RMSE temperatura (°C): {metrics['rmse_temp']:.4f}")

    print("\nMÉTRICA REWARD/RETURN (para tabla):")
    print(f" - Train AvgReturn_Last{int(train_reward_stats['train_return_lastN'])}: {train_reward_stats['train_return_avg_lastN']:.4f}")
    print(f" - Train StdReturn_Last{int(train_reward_stats['train_return_lastN'])}: {train_reward_stats['train_return_std_lastN']:.4f}")
    print(f" - Train BestReturn: {train_reward_stats['train_return_best']:.4f}")
    print(f" - Eval Return Mean (repeats={int(eval_reward_stats['eval_return_repeats'])}): {eval_reward_stats['eval_return_mean']:.4f}")
    print(f" - Eval Return Std: {eval_reward_stats['eval_return_std']:.4f}")

    # CSV
    metrics_path = os.path.join(OUT_DIR, "resultado_ppo.csv")
    pd.DataFrame([{
        "csv_path": CSV_PATH,
        "freq_name": env.freq_name,
        "minutes_per_step": env.minutes_per_step,
        "seed": SEED,
        "fixed_start_step": FIXED_START_STEP,
        "episodes": EPISODES,
        "gamma": cfg.gamma,
        "gae_lambda": cfg.gae_lambda,
        "lr": cfg.lr,
        "clip_eps": cfg.clip_eps,
        "ent_coef": cfg.ent_coef,
        "vf_coef": cfg.vf_coef,
        "update_epochs": cfg.update_epochs,
        "minibatch_size": cfg.minibatch_size,

    
        **metrics,

        
        **train_reward_stats,

       
        **eval_reward_stats,

       
        "eval_return_single": float(eval_return),
        "return_lastN_used": int(RETURN_LAST_N),
    }]).to_csv(metrics_path, index=False)
    print(f"\nCSV guardado: {metrics_path}")

    # Plots
    plot_learning_curve(returns, os.path.join(OUT_DIR, "curva_aprendizaje_rewards_ppo.png"))
    plot_energy(env, real_tap, algo_tap, metrics["savings_pct"], os.path.join(OUT_DIR, "consumo_energetico_ppo.png"))
    plot_comfort(env, temp, os.path.join(OUT_DIR, "control_confort_ppo.png"))
    plot_temp_rmse(env, temp, metrics["rmse_temp"], os.path.join(OUT_DIR, "desempeno_control_termico_rmse_ppo.png"))
    plot_robustness(env, model, device, num_episodes=ROBUSTNESS_EPISODES, out_path=os.path.join(OUT_DIR, "robustez_modelo_ppo.png"))

    print(f"Gráficos guardados en: {OUT_DIR}")


if __name__ == "__main__":
    main()

