import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


folder_path = r"C:\Users\rober\Documents\TESIS\Datasets\Dataset15min" 
timestamp_col = "Local Time Stamp"
tap_col = "Act P (kW)"  
save_prefix = "S2_1h"  


ONLY_DAY = None  

DAY_MAP = {
    "lunes": "Lunes",
    "martes": "Martes",
    "miercoles": "Miércoles",
    "miércoles": "Miércoles",
    "jueves": "Jueves",
    "viernes": "Viernes",
}

def detect_day_from_filename(filename: str) -> str:
    name = os.path.basename(filename).lower()
    for k, v in DAY_MAP.items():
        if k in name:
            return v
    return os.path.splitext(os.path.basename(filename))[0]

def load_scenario(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {folder}")

    frames = []
    for f in files:
        df = pd.read_csv(f)

        if timestamp_col not in df.columns:
            raise ValueError(f"Falta columna '{timestamp_col}' en {os.path.basename(f)}")
        if tap_col not in df.columns:
            raise ValueError(f"Falta columna '{tap_col}' en {os.path.basename(f)}")

        df = df[[timestamp_col, tap_col]].copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df[tap_col] = pd.to_numeric(df[tap_col], errors="coerce")
        df = df.dropna(subset=[timestamp_col, tap_col])

        df["Dia"] = detect_day_from_filename(f)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    return all_df

df = load_scenario(folder_path)

# Variables auxiliares
df["Fecha"] = df[timestamp_col].dt.date
df["Hora"] = df[timestamp_col].dt.hour
df["Minuto"] = df[timestamp_col].dt.minute
df["HoraDecimal"] = df["Hora"] + df["Minuto"]/60.0 

days_order = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
available_days = [d for d in days_order if d in set(df["Dia"])]

#PATRÓN SEMANAL

data_by_day = [df.loc[df["Dia"] == d, tap_col].values for d in available_days]

plt.figure(figsize=(10, 5))
plt.boxplot(data_by_day, labels=available_days, showfliers=True)
plt.ylabel("TAP (kW)")
plt.title("Patrón semanal de TAP (distribución por día)")
plt.tight_layout()
plt.savefig(f"{save_prefix}_patron_semanal_boxplot.png", dpi=300)
plt.show()


#PATRÓN DIARIO

days_to_plot = [ONLY_DAY] if ONLY_DAY else available_days

for day_name in days_to_plot:
    sub = df[df["Dia"] == day_name].copy()

    
    piv = sub.pivot_table(
        index="Fecha",
        columns="HoraDecimal",
        values=tap_col,
        aggfunc="mean"
    ).sort_index(axis=1)

    plt.figure(figsize=(10, 5))
    
    for idx in piv.index:
        y = piv.loc[idx].values
        x = piv.columns.values
        plt.plot(x, y, linewidth=0.8, alpha=0.8)

    plt.xlabel("Hora del día")
    plt.ylabel("TAP (kW)")
    plt.title(f"Patrón diario de TAP (curvas por día){day_name}")
    plt.xticks(np.arange(0, 24, 2))
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_patron_diario_spaghetti_{day_name}.png", dpi=300)
    plt.show()

print(" Imagnees generadas correctamente.")
