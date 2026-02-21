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


# =========================
# CONFIG
# =========================
NOMBRE_ARCHIVO = "Martes.csv"
SEED = 42

FIXED_START_STEP = 0
FORCE_CPU = False

# Entorno / Reward
TARGET_TEMP = 23.0
COMFORT_RANGE = 1.5
W_E = 0.6
W_C = 0.4
POWER_ACTION_SCALE = 0.6

# PPO para evaluar combinaciones (grid search)
GRID_TRAIN_EPISODES = 40
EVAL_EPISODES = 1

# PPO FIXED (no se buscan)
FIXED_GAE_LAMBDA = 0.95
FIXED_UPDATE_EPOCHS = 10
FIXED_MINIBATCH_SIZE = 256
FIXED_HIDDEN = 256
FIXED_VF_COEF = 0.5
FIXED_MAX_GRAD_NORM = 0.5

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, NOMBRE_ARCHIVO)
OUT_DIR = BASE_DIR
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# SEED
# =========================
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


# =========================
# DATA LOAD
# =========================
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


# =========================
# ENV
# =========================
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

        # Reward (igual lógica que PPO.py)
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
        }
        return next_obs, float(reward), terminated, False, info


# =========================
# PPO MODEL
# =========================
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

    def forward(self, x: torch.Tensor):
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
            a_clip = torch.clamp(a, -1 + eps, 1 - eps)
            z = 0.5 * torch.log((1 + a_clip) / (1 - a_clip))

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


# =========================
# METRICS
# =========================
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


# =========================
# PPO TRAIN (corto) + EVAL
# =========================
def train_ppo_short(env: BuildingEnergyEnv, cfg: PPOConfig, device: torch.device, episodes: int):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCritic(state_dim, action_dim, hidden=cfg.hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    returns: List[float] = []

    for ep in range(episodes):
        s, _ = env.reset(seed=SEED)
        buf = RolloutBuffer(env.episode_length, state_dim, action_dim, device)

        ep_return = 0.0

        for t in range(env.episode_length):
            st = torch.tensor(s, dtype=torch.float32, device=device)
            with torch.no_grad():
                a, logp, ent, v = model.get_action_and_value(st.unsqueeze(0))
            a_np = a.squeeze(0).cpu().numpy()

            ns, r, term, trunc, info = env.step(a_np)
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
            for mb_s, mb_a, mb_logp_old, mb_adv, mb_ret, mb_v_old in buf.get_minibatches(cfg.minibatch_size):
                new_a, new_logp, entropy, new_v = model.get_action_and_value(mb_s, action=mb_a)

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

        returns.append(ep_return)

    return model, returns


def eval_one_episode_fixed(env: BuildingEnergyEnv, model: ActorCritic, device: torch.device):
    old_fixed = env.fixed_start_step
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

    env.fixed_start_step = old_fixed  # restaura
    return np.array(real_tap), np.array(algo_tap), np.array(temps)


# =========================
# GRID SEARCH PPO
# Cost = energy_ratio + 2.0 * viol_ratio   (menor es mejor)
# =========================
def eval_config_ppo(df: pd.DataFrame, cfg: PPOConfig, device: torch.device):
    env = BuildingEnergyEnv(
        df,
        target_temp=TARGET_TEMP,
        comfort_range=COMFORT_RANGE,
        w_e=W_E,
        w_c=W_C,
        power_action_scale=POWER_ACTION_SCALE,
        seed=SEED,
        fixed_start_step=None,  # entreno con start aleatorio
    )

    model, returns = train_ppo_short(env, cfg, device, episodes=GRID_TRAIN_EPISODES)

    # Evaluación fija (misma ventana)
    env_eval = BuildingEnergyEnv(
        df,
        target_temp=TARGET_TEMP,
        comfort_range=COMFORT_RANGE,
        w_e=W_E,
        w_c=W_C,
        power_action_scale=POWER_ACTION_SCALE,
        seed=SEED,
        fixed_start_step=FIXED_START_STEP,
    )
    real_tap, algo_tap, temp = eval_one_episode_fixed(env_eval, model, device)
    m = compute_metrics(env_eval, real_tap, algo_tap, temp)

    energy_ratio = m["kwh_algo"] / max(m["kwh_real"], 1e-9)  # <1 mejor
    viol_ratio = (m["viol_pct"] / 100.0)                     # 0..1

    raw_cost = energy_ratio + (2.0 * viol_ratio)

    return {
        "energy_ratio": float(energy_ratio),
        "viol_pct": float(m["viol_pct"]),
        "reward_mean": float(np.mean(returns[-10:])) if len(returns) >= 10 else float(np.mean(returns)),
        "rmse_temp": float(m["rmse_temp"]),
        "raw_cost": float(raw_cost),
        "kwh_real": float(m["kwh_real"]),
        "kwh_algo": float(m["kwh_algo"]),
        "savings_pct": float(m["savings_pct"]),
    }


# =========================
# MAIN GRID 
# =========================
if __name__ == "__main__":
    set_seed(SEED)

    if FORCE_CPU:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device}")

    print(f"CSV: {CSV_PATH}")
    df = load_and_prepare(CSV_PATH)

    # ---- SEARCH SPACE (SOLO 4 hiperparámetros principales = 81)
    search_space = {
        "gamma":    [0.90, 0.95, 0.99],
        "lr":       [0.0001, 0.0003, 0.001],
        "clip_eps": [0.1, 0.2, 0.3],
        "ent_coef": [0.0, 0.01, 0.05],
    }

    total = 1
    for k in search_space:
        total *= len(search_space[k])

    print(f"Total combinaciones: {total} (debería ser 81)")

    results = []
    count = 0
    start_time = time.time()

    for gamma in search_space["gamma"]:
        for lr in search_space["lr"]:
            for clip_eps in search_space["clip_eps"]:
                for ent_coef in search_space["ent_coef"]:
                    count += 1
                    print(f"\n[{count}/{total}] g={gamma}, lr={lr}, clip={clip_eps}, ent={ent_coef} ...", end="")

                    try:
                        cfg = PPOConfig(
                            gamma=gamma,
                            gae_lambda=FIXED_GAE_LAMBDA,
                            lr=lr,
                            clip_eps=clip_eps,
                            ent_coef=ent_coef,
                            vf_coef=FIXED_VF_COEF,
                            max_grad_norm=FIXED_MAX_GRAD_NORM,
                            update_epochs=FIXED_UPDATE_EPOCHS,
                            minibatch_size=FIXED_MINIBATCH_SIZE,
                            hidden=FIXED_HIDDEN,
                        )

                        m = eval_config_ppo(df, cfg, device)

                        results.append({
                            "gamma": gamma,
                            "lr": lr,
                            "clip_eps": clip_eps,
                            "ent_coef": ent_coef,
                            # fijos (se guardan para trazabilidad)
                            "gae_lambda": cfg.gae_lambda,
                            "vf_coef": cfg.vf_coef,
                            "max_grad_norm": cfg.max_grad_norm,
                            "update_epochs": cfg.update_epochs,
                            "minibatch_size": cfg.minibatch_size,
                            "hidden": cfg.hidden,
                            **m
                        })

                        print(f" energy_ratio={m['energy_ratio']:.3f} "
                              f"viol={m['viol_pct']:.2f}% "
                              f"rmse={m['rmse_temp']:.3f} "
                              f"cost={m['raw_cost']:.3f}", end="")

                    except Exception as e:
                        print(f" ERROR: {e}")

    df_res = pd.DataFrame(results)
    if len(df_res) == 0:
        raise RuntimeError("No se generaron resultados. Revisa el CSV o el entorno.")

    # Normalizar costo 0..1 (menor mejor)
    cmin, cmax = df_res["raw_cost"].min(), df_res["raw_cost"].max()
    df_res["score_0_1"] = (df_res["raw_cost"] - cmin) / max((cmax - cmin), 1e-9)

    df_res = df_res.sort_values("score_0_1", ascending=True)

    out_csv = f"grid_results_ppo_{os.path.splitext(NOMBRE_ARCHIVO)[0]}.csv"
    df_res.to_csv(out_csv, index=False)

    best = df_res.iloc[0]
    mins = (time.time() - start_time) / 60.0

    print("\n" + "=" * 70)
    print(f"TERMINADO en {mins:.1f} min | Mejor configuración (MENOR score):")
    print(best[[
        "gamma", "lr", "clip_eps", "ent_coef",
        "energy_ratio", "viol_pct", "rmse_temp", "raw_cost", "score_0_1"
    ]])
    print(f"CSV guardado: {out_csv}")
    print("=" * 70)

    # Gráfico Top 10
    top10 = df_res.head(10).copy()
    labels = [
        f"g={r.gamma},lr={r.lr},c={r.clip_eps},e={r.ent_coef}"
        for r in top10.itertuples(index=False)
    ]
    vals = top10["score_0_1"].values

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(vals)), vals)
    plt.ylim(0, 1.0)
    plt.title("Top 10 Configuraciones PPO (Score 0–1, menor es mejor)")
    plt.ylabel("Score (0..1)")
    plt.xlabel("Combinación de Hiperparámetros")
    plt.xticks(range(len(vals)), labels, rotation=35, ha="right")

    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    out_png = f"grid_top10_ppo_{os.path.splitext(NOMBRE_ARCHIVO)[0]}.png"
    plt.savefig(out_png, dpi=200)
    print(f"PNG guardado: {out_png}")
    plt.show()
