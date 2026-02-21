import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
import os, time
import matplotlib.pyplot as plt


NOMBRE_ARCHIVO = "viernes_hora.csv"   # Cambiar por el archivo deseado
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando Hardware: {DEVICE}")


# CARGA + PREPROCESO
if not os.path.exists(NOMBRE_ARCHIVO):
    raise FileNotFoundError(f"No encuentro '{NOMBRE_ARCHIVO}'.")

df_raw = pd.read_csv(NOMBRE_ARCHIVO)

col_map = {
    "Act P (kW)": "TAP",
    "TempR1 (C)": "TF1",
    "TempR2 (C)": "TF2",
    "TempR3 (C)": "TF3",
    "HUMEDADR1 (%rH)": "HF1",
    "Local Time Stamp": "Timestamp",
}
df = df_raw.rename(columns=col_map).copy()
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df["hour"] = df["Timestamp"].dt.hour
df["minute"] = df["Timestamp"].dt.minute
time_mins = df["hour"] * 60 + df["minute"]
df["hour_sin"] = np.sin(2 * np.pi * time_mins / (24 * 60))
df["hour_cos"] = np.cos(2 * np.pi * time_mins / (24 * 60))

df = df.fillna(method="ffill").fillna(method="bfill")

# ENV 
class BuildingEnergyEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)

        time_delta = self.df["Timestamp"].iloc[1] - self.df["Timestamp"].iloc[0]
        minutes = time_delta.total_seconds() / 60

        if 13 <= minutes <= 17:
            self.episode_length = 96
            self.freq_name = "15 minutos"
            self.temp_inertia = 1.5
        elif 55 <= minutes <= 65:
            self.episode_length = 24
            self.freq_name = "1 Hora"
            self.temp_inertia = 2.5
        else:
            print(f"Alerta: Frecuencia extraña ({minutes} min). Asumo 15 min.")
            self.episode_length = 96
            self.freq_name = "Desconocida (Default 96 pasos)"
            self.temp_inertia = 1.5

        print(f"Entorno: {self.freq_name}. Pasos/episodio: {self.episode_length}")

        self.start_step = 0
        self.max_power = float(self.df["TAP"].max())

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        self.target_temp = 23.0
        self.comfort_range = 1.5

    def reset(self, seed=None):
        super().reset(seed=seed)
        max_idx = len(self.df) - self.episode_length - 1
        self.start_step = np.random.randint(0, max_idx)
        self.current_step_index = 0
        return self._get_obs(self.start_step), {}

    def _get_obs(self, step_idx):
        row = self.df.iloc[step_idx]
        return np.array([
            row["TAP"], row["TF1"], row["TF2"], row["TF3"], row["HF1"],
            row["hour_sin"], row["hour_cos"]
        ], dtype=np.float32)

    def step(self, action):
        act_val = float(action[0])
        real_idx = self.start_step + self.current_step_index

        next_row = self.df.iloc[real_idx + 1]
        base_tap = float(next_row["TAP"])
        base_temp_avg = float((next_row["TF1"] + next_row["TF2"] + next_row["TF3"]) / 3)

        power_mod = 1.0 + (act_val * 0.6)
        temp_mod = -(act_val * self.temp_inertia)

        sim_tap = max(0.0, base_tap * power_mod)
        sim_temp = base_temp_avg + temp_mod

        # Recompensa
        r_energy = -(sim_tap / self.max_power)
        dist = abs(sim_temp - self.target_temp)
        if dist <= self.comfort_range:
            r_comfort = 1.0
        else:
            r_comfort = -(dist * 2.0)

        reward = (1.5 * r_energy) + r_comfort

        self.current_step_index += 1
        terminated = self.current_step_index >= self.episode_length

        next_obs = self._get_obs(real_idx + 1)
        next_obs[0] = sim_tap
        next_obs[1] += temp_mod
        next_obs[2] += temp_mod
        next_obs[3] += temp_mod

        info = {"real_tap": base_tap, "sac_tap": sim_tap, "sim_temp": sim_temp}
        return next_obs, float(reward), terminated, False, info

# REPLAY BUFFER
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.cap = capacity
        self.ptr = 0
        self.size = 0
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity, action_dim), dtype=np.float32)
        self.r = np.zeros((capacity, 1), dtype=np.float32)
        self.ns = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, s, a, r, ns, d):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.ns[self.ptr] = ns
        self.d[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.s[idx]).to(DEVICE),
            torch.tensor(self.a[idx]).to(DEVICE),
            torch.tensor(self.r[idx]).to(DEVICE),
            torch.tensor(self.ns[idx]).to(DEVICE),
            torch.tensor(self.d[idx]).to(DEVICE),
        )


LOG_STD_MIN, LOG_STD_MAX = -20, 2

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, s):
        x = self.backbone(s)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, s):
        mean, log_std = self(s)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()               # reparameterization trick
        a_t = torch.tanh(x_t)                # squash a [-1,1]

        # log_prob con corrección por tanh
        log_prob = normal.log_prob(x_t).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - a_t.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return a_t, log_prob, torch.tanh(mean)

class SACReal:
    def __init__(self, state_dim, action_dim, lr, gamma, tau, alpha):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.q1 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q2 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q1_t = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q2_t = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        self.pi = GaussianPolicy(state_dim, action_dim).to(DEVICE)

        self.q_opt = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.pi_opt = optim.Adam(self.pi.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, state, deterministic=False):
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        if deterministic:
            _, _, a = self.pi.sample(s)
            return a.cpu().numpy()[0]
        else:
            a, _, _ = self.pi.sample(s)
            return a.cpu().numpy()[0]

    def soft_update(self, net, net_t):
        for p, pt in zip(net.parameters(), net_t.parameters()):
            pt.data.copy_(pt.data * (1 - self.tau) + p.data * self.tau)

    def update(self, rb, batch_size=64):
        s, a, r, ns, d = rb.sample(batch_size)

        with torch.no_grad():
            na, nlogp, _ = self.pi.sample(ns)
            q1n = self.q1_t(ns, na)
            q2n = self.q2_t(ns, na)
            qn = torch.min(q1n, q2n) - self.alpha * nlogp
            target = r + (1 - d) * self.gamma * qn

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # Policy loss
        a_pi, logp_pi, _ = self.pi.sample(s)
        q1_pi = self.q1(s, a_pi)
        q2_pi = self.q2(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        pi_loss = (self.alpha * logp_pi - q_pi).mean()

        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        self.soft_update(self.q1, self.q1_t)
        self.soft_update(self.q2, self.q2_t)

# Score = (energía relativa vs baseline) + penalización por violaciones
def eval_config(env, agent, episodes=3, warmup_steps=200, train_updates=True):
    rb = ReplayBuffer(10000, env.observation_space.shape[0], env.action_space.shape[0])

    minutes = (env.df["Timestamp"].iloc[1] - env.df["Timestamp"].iloc[0]).total_seconds() / 60.0
    minutes_per_step = 15 if minutes < 30 else 60

    total_energy_ratio = []
    total_viol_pct = []
    total_reward = []

    for ep in range(episodes):
        s, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        viol = 0

        sum_base_kwh = 0.0
        sum_sac_kwh = 0.0

        while not done:
            # acción
            a = agent.act(s, deterministic=False)
            ns, r, done, _, info = env.step(a)

            rb.add(s, a, [r], ns, [float(done)])
            s = ns
            ep_reward += float(r)

            # kWh aproximado por paso (TAP es kW)
            sum_base_kwh += float(info["real_tap"]) * (minutes_per_step / 60.0)
            sum_sac_kwh  += float(info["sac_tap"])  * (minutes_per_step / 60.0)

            # violaciones confort
            if info["sim_temp"] < (env.target_temp - env.comfort_range) or info["sim_temp"] > (env.target_temp + env.comfort_range):
                viol += 1

            steps += 1

            # update
            if train_updates and rb.size > 256 and steps % 10 == 0:
                agent.update(rb, batch_size=64)

            # guardia
            if steps > env.episode_length + 5:
                done = True

        energy_ratio = (sum_sac_kwh / max(sum_base_kwh, 1e-6))  # <1 es mejor (ahorro)
        viol_pct = (viol / max(steps, 1))                        # 0..1

        total_energy_ratio.append(energy_ratio)
        total_viol_pct.append(viol_pct)
        total_reward.append(ep_reward)

    return {
        "energy_ratio": float(np.mean(total_energy_ratio)),
        "viol_pct": float(np.mean(total_viol_pct)),
        "reward_mean": float(np.mean(total_reward)),
    }


# GRID SEARCH

if __name__ == "__main__":
    env0 = BuildingEnergyEnv(df)

    search_space = {
        "gamma": [0.90, 0.95, 0.99],
        "alpha": [0.05, 0.1, 0.2],
        "tau":   [0.005, 0.01, 0.02],
        "lr":    [0.0001, 0.0003, 0.001],
    }

    results = []
    total = len(search_space["gamma"]) * len(search_space["alpha"]) * len(search_space["tau"]) * len(search_space["lr"])
    count = 0
    start_time = time.time()

    for g in search_space["gamma"]:
        for a in search_space["alpha"]:
            for t in search_space["tau"]:
                for l in search_space["lr"]:
                    count += 1
                    print(f"\n[{count}/{total}] gamma={g}, alpha={a}, tau={t}, lr={l} ...", end="")

                    try:
                        env = BuildingEnergyEnv(df)
                        agent = SACReal(
                            state_dim=env.observation_space.shape[0],
                            action_dim=env.action_space.shape[0],
                            lr=l, gamma=g, tau=t, alpha=a
                        )

                        m = eval_config(env, agent, episodes=3)

                        # COSTO crudo: energía relativa + peso * violaciones
                        # (menor es mejor)
                        raw_cost = m["energy_ratio"] + (2.0 * m["viol_pct"])

                        results.append({
                            "gamma": g, "alpha": a, "tau": t, "lr": l,
                            "energy_ratio": m["energy_ratio"],
                            "viol_pct": m["viol_pct"],
                            "reward_mean": m["reward_mean"],
                            "raw_cost": raw_cost
                        })

                        print(f" energy_ratio={m['energy_ratio']:.3f} viol={m['viol_pct']:.3f} cost={raw_cost:.3f}", end="")

                    except Exception as e:
                        print(f" ERROR: {e}")

    df_res = pd.DataFrame(results)

    # Normalización 0..1 del COSTO (menor mejor)
    cmin, cmax = df_res["raw_cost"].min(), df_res["raw_cost"].max()
    df_res["score_0_1"] = (df_res["raw_cost"] - cmin) / max((cmax - cmin), 1e-9)

    # Ordenamos: menor score es mejor
    df_res = df_res.sort_values("score_0_1", ascending=True)

    out_csv = f"grid_results_{os.path.splitext(NOMBRE_ARCHIVO)[0]}.csv"
    df_res.to_csv(out_csv, index=False)

    best = df_res.iloc[0]
    mins = (time.time() - start_time) / 60

    print("\n" + "="*60)
    print(f"TERMINADO en {mins:.1f} min | Mejor configuración (MENOR score):")
    print(best[["gamma","alpha","tau","lr","energy_ratio","viol_pct","raw_cost","score_0_1"]])
    print(f"CSV guardado: {out_csv}")
    print("="*60)

    # Gráfico Top 10
    top10 = df_res.head(10).copy()
    labels = [
        f"G={r.gamma},T={r.tau},A={r.alpha},L={r.lr}"
        for r in top10.itertuples(index=False)
    ]
    vals = top10["score_0_1"].values

    plt.figure(figsize=(14,6))
    plt.bar(range(len(vals)), vals)
    plt.ylim(0, 1.0)
    plt.title("Top 10 Configuraciones de Hiperparámetros (Score 0–1, menor es mejor)")
    plt.ylabel("Score (0..1)")
    plt.xlabel("Combinación de Hiperparámetros")
    plt.xticks(range(len(vals)), labels, rotation=35, ha="right")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()

    out_png = f"grid_top10_{os.path.splitext(NOMBRE_ARCHIVO)[0]}.png"
    plt.savefig(out_png, dpi=200)
    print(f"PNG guardado: {out_png}")
    plt.show()
