#!/usr/bin/env python
# -----------------------------------------------------------------------------
#  PINO_teslaS_physics_test.py — Physics-only evaluator for Tesla Model S
# -----------------------------------------------------------------------------
import argparse
import numpy as np, pandas as pd, scipy.signal
from pathlib import Path
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch

# ---- constants & bounds (Tesla Model S) ----
RHO, A, G = 1.17, 2.40, 9.81     # air density, frontal area (S≈2.40 m²), gravity
CD_L, CD_H     = 0.21 , 0.26     # aerodynamic drag (tight, S-specific)
CRR_L,CRR_H    = 0.005, 0.012    # rolling resistance
M_L,  M_H      = 2200., 2400.    # mass bounds (heavier than M3)
PA_L, PA_H     = 0., 1000.    # single constant Paux (idle ~1.39 kW)
ETA_L,ETA_H    = 0.60 , 0.90
ETAR_L,ETAR_H  = 0.20 , 0.90

def sigmoid_bound(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(raw)

def load_scalars(ckpt_path: Path):
    """Return Cd, Crr, m, Paux, eta, etar as floats from checkpoint."""
    ck = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    st = ck["state_dict"]
    if "raw_cd" in st:
        Cd   = float(sigmoid_bound(st["raw_cd"],   CD_L,  CD_H))
        Crr  = float(sigmoid_bound(st["raw_crr"],  CRR_L, CRR_H))
        m    = float(sigmoid_bound(st["raw_m"],    M_L,   M_H))
        Paux = float(sigmoid_bound(st["raw_paux"], PA_L,  PA_H))
        eta  = float(sigmoid_bound(st["raw_eta"],  ETA_L, ETA_H))
        etar = float(sigmoid_bound(st["raw_etar"], ETAR_L,ETAR_H))
    else:  # legacy fallback (log-params)
        Cd   = float(torch.exp(st["log_Cd"]))
        Crr  = float(torch.exp(st["log_Crr"]))
        m    = float(torch.exp(st["log_m"]))
        Paux = float(ck.get("Paux", 1390.0))
        eta  = float(ck.get("eta",  0.80))
        etar = float(ck.get("etar", 0.65))
    return Cd, Crr, m, Paux, eta, etar

def physics_power(v, a, Cd, Crr, m, Paux, eta, etar):
    """Vectorized physics power with regen gate a<-0.045 (matches training)."""
    aero = 0.5 * RHO * A * Cd * v**3
    roll = Crr * m * G * v
    iner = m * a * v
    iner_adj = np.where(a < -0.045, iner * etar, iner)  # regen eff on decel inertial term
    return (aero + roll + iner_adj) / eta + Paux

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Physics-only tester for Tesla Model S (EV-PINO)")
    ap.add_argument("--ckpt", default="pino_teslaS_params.pt",
                    help="Checkpoint saved by EV-PINO trainer (Tesla S)")
    ap.add_argument("--csv",  default="teslaS_SG.csv",
                    help="CSV with columns: Speed_SG, Acceleration_SG, BatteryPower_SG")
    ap.add_argument("--fs", type=float, default=10.0,
                    help="Sampling rate (Hz) for PSD (default: 10)")
    ap.add_argument("--out_dir", default="eval_out_S",
                    help="Directory to save CSVs/figure")
    ap.add_argument("--out_prefix", default="teslaS_physics",
                    help="Filename prefix for saved CSVs/figure")
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
    pd.DataFrame({
        "index": t_idx,
        "Speed_SG_mps": v,
        "Acceleration_SG_mps2": a,
        "P_true_W": P_true,
        "P_pred_W": P_pred,
        "Residual_W": residual
    }).to_csv(ts_csv, index=False)

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
