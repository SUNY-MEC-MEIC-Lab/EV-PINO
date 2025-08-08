#!/usr/bin/env python
# -----------------------------------------------------------------------------
#  PINO_EV9_physics_test.py — EV9 physics-only evaluator (no neural heads)
# -----------------------------------------------------------------------------
import argparse
from pathlib import Path
from math import sqrt
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
from sklearn.preprocessing import StandardScaler  # allow-listed for safe load

# ---- constants (EV9) ----
RHO, A, G = 1.17, 3.05, 9.81  # air density [kg/m^3], frontal area [m^2], gravity [m/s^2]

# bounds used to map raw checkpoint scalars -> physical values
CD_L,   CD_H   = 0.27 , 0.29
CRR_L,  CRR_H  = 0.0055, 0.0075
M_L,    M_H    = 2650., 2880.
PA_L,   PA_H   = 300.,  800.
ETA_L,  ETA_H  = 0.60 , 0.90
ETAR_L, ETAR_H = 0.20 , 0.80

def sigmoid_bound(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(raw)

def load_scalars_from_ckpt(ckpt_path: Path):
    """Return Cd, Crr, m, Paux, eta, etar (floats) from training checkpoint."""
    # Need safe load (PyTorch >=2.6) because StandardScaler is pickled in the file
    with torch.serialization.safe_globals([StandardScaler]):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    st = ck.get("state_dict", {})
    if {"raw_cd","raw_crr","raw_m","raw_paux","raw_eta","raw_etar"} <= set(st.keys()):
        Cd   = float(sigmoid_bound(st["raw_cd"],   CD_L,  CD_H))
        Crr  = float(sigmoid_bound(st["raw_crr"],  CRR_L, CRR_H))
        m    = float(sigmoid_bound(st["raw_m"],    M_L,   M_H))
        Paux = float(sigmoid_bound(st["raw_paux"], PA_L,  PA_H))
        eta  = float(sigmoid_bound(st["raw_eta"],  ETA_L, ETA_H))
        etar = float(sigmoid_bound(st["raw_etar"], ETAR_L,ETAR_H))
        return Cd, Crr, m, Paux, eta, etar

    # Fallbacks if the checkpoint is legacy/minimal
    Cd   = float(ck.get("Cd",   (CD_L+CD_H)/2))
    Crr  = float(ck.get("Crr",  (CRR_L+CRR_H)/2))
    m    = float(ck.get("m",    (M_L+M_H)/2))
    Paux = float(ck.get("Paux", (PA_L+PA_H)/2))
    eta  = float(ck.get("eta",  (ETA_L+ETA_H)/2))
    etar = float(ck.get("etar", (ETAR_L+ETAR_H)/2))
    return Cd, Crr, m, Paux, eta, etar

def physics_power_ev9(v, a, *, Cd, Crr, m, Paux, eta, etar):
    """
    Vectorized physics power (W).
    v [m/s], a [m/s^2] as 1D numpy arrays.
    """
    aero = 0.5 * RHO * A * Cd * v**3
    roll = Crr * m * G * v
    iner = m * a * v
    Pm   = aero + roll + iner
    Ptr  = np.maximum(Pm, 0.0)           # traction (+)
    Prg  = np.minimum(Pm, 0.0)           # regen (−)
    return Ptr/eta + etar*Prg + Paux     # note Prg is negative

def main():
    ap = argparse.ArgumentParser("EV9 physics-only evaluator")
    ap.add_argument("--ckpt", default="pino_ev9_params.pt", help="Training checkpoint to read scalars")
    ap.add_argument("--csv",  default="ev9_train.csv",
                    help="CSV with Speed_SG [m/s], Acceleration_SG [m/s^2], BatteryPower_SG [W]")
    ap.add_argument("--fs", type=float, default=10.0, help="Sampling rate (Hz) for PSD")
    ap.add_argument("--out_dir", default="eval_ev9_phys", help="Directory to save outputs")
    ap.add_argument("--out_prefix", default="ev9_phys", help="Filename prefix")
    # Optional manual overrides (if provided, these take precedence)
    ap.add_argument("--Cd",   type=float)
    ap.add_argument("--Crr",  type=float)
    ap.add_argument("--m",    type=float)
    ap.add_argument("--Paux", type=float)
    ap.add_argument("--eta",  type=float)
    ap.add_argument("--etar", type=float, help="regen efficiency")
    ap.add_argument("--save_fig", action="store_true")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load or override scalars
    Cd0, Crr0, m0, Paux0, eta0, etar0 = load_scalars_from_ckpt(Path(args.ckpt))
    Cd   = args.Cd   if args.Cd   is not None else Cd0
    Crr  = args.Crr  if args.Crr  is not None else Crr0
    m    = args.m    if args.m    is not None else m0
    Paux = args.Paux if args.Paux is not None else Paux0
    eta  = args.eta  if args.eta  is not None else eta0
    etar = args.etar if args.etar is not None else etar0

    print(f"Physics scalars → Cd={Cd:.3f}  Crr={Crr:.4f}  m={m:.0f} kg  "
          f"Paux={Paux:.0f} W  η={eta:.3f}  ηreg={etar:.3f}")

    # Data
    df = pd.read_csv(args.csv)
    v = df["Speed_SG"].to_numpy(np.float32)
    a = df["Acceleration_SG"].to_numpy(np.float32)
    P_true = df["BatteryPower_SG"].to_numpy(np.float32)
    t_idx = np.arange(len(v), dtype=np.int64)

    # Predict (physics only)
    P_pred = physics_power_ev9(v, a, Cd=Cd, Crr=Crr, m=m, Paux=Paux, eta=eta, etar=etar)
    residual = P_true - P_pred

    # Metrics
    rmse = sqrt(mean_squared_error(P_true, P_pred))
    mae  = mean_absolute_error(P_true, P_pred)
    r2   = r2_score(P_true, P_pred)
    mae_d= mean_absolute_error(np.diff(P_true), np.diff(P_pred))
    print(f"RMSE {rmse:,.0f} W  |  MAE {mae:,.0f} W  |  R² {r2:.4f}")
    print(f"MAE on Δ: {mae_d:,.0f} W/step")

    # Save CSVs
    ts_csv = out_dir / f"{args.out_prefix}_timeseries.csv"
    pd.DataFrame({
        "index": t_idx,
        "Speed_SG_mps": v,
        "Acceleration_SG_mps2": a,
        "P_true_W": P_true,
        "P_pred_W": P_pred.astype(np.float32),
        "Residual_W": residual.astype(np.float32)
    }).to_csv(ts_csv, index=False)

    met_csv = out_dir / f"{args.out_prefix}_metrics.csv"
    pd.DataFrame([{
        "ckpt": str(Path(args.ckpt).resolve()),
        "data_csv": str(Path(args.csv).resolve()),
        "n_samples": int(len(v)),
        "RMSE_W": float(rmse),
        "MAE_W": float(mae),
        "R2": float(r2),
        "Delta_MAE_W_per_step": float(mae_d),
        "Cd": float(Cd), "Crr": float(Crr), "m_kg": float(m),
        "Paux_W": float(Paux), "eta": float(eta), "eta_reg": float(etar),
    }]).to_csv(met_csv, index=False)

    # PSDs
    nper = int(min(4096, len(P_true)))
    f, psd_true = scipy.signal.welch(P_true, fs=args.fs, nperseg=nper)
    _, psd_pred = scipy.signal.welch(P_pred, fs=args.fs, nperseg=nper)
    psd_csv = out_dir / f"{args.out_prefix}_psd.csv"
    pd.DataFrame({"freq_Hz": f, "PSD_true": psd_true, "PSD_pred": psd_pred}).to_csv(psd_csv, index=False)

    print(f"saved CSVs → {ts_csv.name}, {met_csv.name}, {psd_csv.name} (in {out_dir.resolve()})")

    # Plots
    if not args.no_plot:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(4, 1, figsize=(12, 18), tight_layout=True)

        ax[0].plot(t_idx, P_true, label="Ground Truth", lw=1)
        ax[0].plot(t_idx, P_pred, "--", label="Physics model", lw=1)
        ax[0].set_title("Battery Power — Physics-only"); ax[0].legend()

        ax[1].scatter(P_true, P_pred, s=5, alpha=.5)
        lo, hi = float(min(P_true.min(), P_pred.min())), float(max(P_true.max(), P_pred.max()))
        ax[1].plot([lo, hi], [lo, hi], 'k--'); ax[1].axis('equal')
        ax[1].set_title("Parity")

        ax[2].scatter(t_idx, residual, s=5, alpha=.5, c='g'); ax[2].axhline(0, ls='--', c='k')
        ax[2].set_title("Residuals (True - Physics)")

        ax[3].semilogy(f, psd_true, label="Ground Truth")
        ax[3].semilogy(f, psd_pred, '--', label="Physics model")
        ax[3].set_xlabel("Frequency (Hz)"); ax[3].set_title("PSD"); ax[3].legend()

        if args.save_fig:
            fig_path = out_dir / f"{args.out_prefix}_plots.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print("saved figure →", fig_path)
        plt.show()

if __name__ == "__main__":
    main()
