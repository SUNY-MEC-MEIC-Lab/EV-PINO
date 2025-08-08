#!/usr/bin/env python
# -----------------------------------------------------------------------------
#  PINO_tesla3_physics_test.py — Physics-only evaluator for Tesla Model 3
# -----------------------------------------------------------------------------
import argparse, os
import numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy.signal
from pathlib import Path
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch

# ---- constants & bounds (match EVPINO Tesla3 train) ----
RHO, A, G = 1.17, 2.22, 9.81
CD_L,CD_H     = 0.23 , 0.25
CRR_L,CRR_H   = 0.009, 0.011
M_L,  M_H     = 1950., 2000.
PA_L, PA_H    = 1000., 1200.
ETA_L,ETA_H   = 0.60 , 0.90
ETAR_L,ETAR_H = 0.20 , 0.90

def sigmoid_bound(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(raw)

def load_scalars(ckpt_path: Path):
    """Return Cd, Crr, m, Paux, eta, etar as floats from checkpoint."""
    ck = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    st = ck["state_dict"]
    # new-style bounded scalars
    if "raw_cd" in st:
        Cd   = float(sigmoid_bound(st["raw_cd"],   CD_L,  CD_H))
        Crr  = float(sigmoid_bound(st["raw_crr"],  CRR_L, CRR_H))
        m    = float(sigmoid_bound(st["raw_m"],    M_L,   M_H))
        Paux = float(sigmoid_bound(st["raw_paux"], PA_L,  PA_H))
        eta  = float(sigmoid_bound(st["raw_eta"],  ETA_L, ETA_H))
        etar = float(sigmoid_bound(st["raw_etar"], ETAR_L,ETAR_H))
    else:  # legacy (log-params)
        Cd   = float(torch.exp(st["log_Cd"]))
        Crr  = float(torch.exp(st["log_Crr"]))
        m    = float(torch.exp(st["log_m"]))
        Paux = float(ck.get("Paux", 1100.0))
        eta  = float(ck.get("eta",  0.80))
        etar = float(ck.get("etar", 0.65))
    return Cd, Crr, m, Paux, eta, etar

def physics_power(v, a, Cd, Crr, m, Paux, eta, etar):
    """Vectorized physics power with regen gate a<-0.045 (matches training)."""
    aero = 0.5 * RHO * A * Cd * v**3
    roll = Crr * m * G * v
    iner = m * a * v
    iner_adj = np.where(a < -0.045, iner * etar, iner)  # regen efficiency on decel inertial term
    return (aero + roll + iner_adj) / eta + Paux

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Physics-only tester for Tesla Model 3 (EVPINO)")
    ap.add_argument("--ckpt", default="pino_ev_params.pt",
                    help="Checkpoint saved by EVPINO physics-only trainer")
    ap.add_argument("--csv",  default="Tesla3_JW_SG.csv",
                    help="CSV with columns: Speed_SG, Acceleration_SG, BatteryPower_SG")
    ap.add_argument("--fs", type=float, default=10.0,
                    help="Sampling rate (Hz) for PSD (default: 10)")
    ap.add_argument("--out_dir", default="eval_out",
                    help="Directory to save CSVs/figure")
    ap.add_argument("--out_prefix", default="tesla3_physics",
                    help="Filename prefix for saved CSVs/figure")
    ap.add_argument("--save_fig", action="store_true", help="Save the figure PNG")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- load scalars ---
    Cd, Crr, m, Paux, eta, etar = load_scalars(args.ckpt)
    print(f"Loaded  Cd={Cd:.3f}  Crr={Crr:.4f}  m={m:.0f} kg  Paux={Paux:.0f} W  η={eta:.3f}  ηreg={etar:.3f}")

    # --- data ---
    df = pd.read_csv(args.csv)
    v = df["Speed_SG"].to_numpy(np.float32)
    a = df["Acceleration_SG"].to_numpy(np.float32)
    P_true = df["BatteryPower_SG"].to_numpy(np.float32)
    t_idx = np.arange(len(v), dtype=np.int64)

    # --- prediction ---
    P_pred = physics_power(v, a, Cd, Crr, m, Paux, eta, etar)
    residual = P_true - P_pred

    # --- metrics ---
    rmse = sqrt(mean_squared_error(P_true, P_pred))
    mae  = mean_absolute_error(P_true, P_pred)
    r2   = r2_score(P_true, P_pred)
    mae_d= mean_absolute_error(np.diff(P_true), np.diff(P_pred))
    print(f"RMSE {rmse:,.0f} W  |  MAE {mae:,.0f} W  |  R² {r2:.4f}")
    print(f"MAE on Δ: {mae_d:,.0f} W/step")

    # --- save CSVs ---
    # 1) Timeseries
    ts_csv = out_dir / f"{args.out_prefix}_timeseries.csv"
    ts_df = pd.DataFrame({
        "index": t_idx,
        "Speed_SG_mps": v,
        "Acceleration_SG_mps2": a,
        "P_true_W": P_true,
        "P_pred_W": P_pred,
        "Residual_W": residual
    })
    ts_df.to_csv(ts_csv, index=False)

    # 2) Metrics (single row)
    met_csv = out_dir / f"{args.out_prefix}_metrics.csv"
    pd.DataFrame([{
        "ckpt": str(Path(args.ckpt).resolve()),
        "data_csv": str(Path(args.csv).resolve()),
        "n_samples": int(len(v)),
        "RMSE_W": float(rmse),
        "MAE_W": float(mae),
        "R2": float(r2),
        "Delta_MAE_W_per_step": float(mae_d),
        "Cd": float(Cd),
        "Crr": float(Crr),
        "m_kg": float(m),
        "Paux_W": float(Paux),
        "eta": float(eta),
        "eta_reg": float(etar)
    }]).to_csv(met_csv, index=False)

    # 3) PSD (Welch)
    nper = int(min(4096, len(P_true)))
    f, psd_true = scipy.signal.welch(P_true, fs=args.fs, nperseg=nper)
    _, psd_pred = scipy.signal.welch(P_pred, fs=args.fs, nperseg=nper)
    psd_csv = out_dir / f"{args.out_prefix}_psd.csv"
    pd.DataFrame({"freq_Hz": f, "PSD_true": psd_true, "PSD_pred": psd_pred}).to_csv(psd_csv, index=False)

    print(f"saved CSVs → {ts_csv.name}, {met_csv.name}, {psd_csv.name} (in {out_dir.resolve()})")

    # --- plots ---
    if not args.no_plot:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(4, 1, figsize=(12, 18), tight_layout=True)

        ax[0].plot(t_idx, P_true, label="Ground Truth")
        ax[0].plot(t_idx, P_pred, "--", label="EV-PINO")
        ax[0].set_title("Battery Power (Time series)"); ax[0].legend()

        ax[1].scatter(P_true, P_pred, s=5, alpha=.5)
        lo, hi = min(P_true.min(), P_pred.min()), max(P_true.max(), P_pred.max())
        ax[1].plot([lo, hi], [lo, hi], 'k--'); ax[1].axis('equal')
        ax[1].set_title("Parity plot")

        ax[2].scatter(t_idx, residual, s=5, alpha=.5, c='g'); ax[2].axhline(0, ls='--', c='k')
        ax[2].set_title("Residuals (True - Pred)")

        ax[3].semilogy(f, psd_true, label="Ground Truth")
        ax[3].semilogy(f, psd_pred, '--', label="EV-PINO")
        ax[3].set_xlabel("Frequency (Hz)"); ax[3].set_title("PSD comparison"); ax[3].legend()

        if args.save_fig:
            fig_path = out_dir / f"{args.out_prefix}_plots.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print("saved figure →", fig_path)
        plt.show()
