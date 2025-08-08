import argparse
from pathlib import Path
from math import sqrt
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
from sklearn.preprocessing import StandardScaler
from utils import VEHICLE_CONFIGS, sigmoid_bound

def load_scalars_from_ckpt(ckpt_path: Path, vehicle: str) -> tuple[float, ...]:
    """Return Cd, Crr, m, Paux, eta, etar (floats) from training checkpoint."""
    config = VEHICLE_CONFIGS[vehicle]
    
    # Need safe load (PyTorch >=2.6) because StandardScaler is pickled in the file
    with torch.serialization.safe_globals([StandardScaler]):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    st = ck.get("state_dict", {})
    if {"raw_cd", "raw_crr", "raw_m", "raw_paux", "raw_eta", "raw_etar"} <= set(st.keys()):
        Cd = float(sigmoid_bound(st["raw_cd"], config['CD_L'], config['CD_H']))
        Crr = float(sigmoid_bound(st["raw_crr"], config['CRR_L'], config['CRR_H']))
        m = float(sigmoid_bound(st["raw_m"], config['M_L'], config['M_H']))
        Paux = float(sigmoid_bound(st["raw_paux"], config['PA_L'], config['PA_H']))
        eta = float(sigmoid_bound(st["raw_eta"], config['ETA_L'], config['ETA_H']))
        etar = float(sigmoid_bound(st["raw_etar"], config['ETAR_L'], config['ETAR_H']))
        return Cd, Crr, m, Paux, eta, etar

    # Fallbacks if the checkpoint is legacy/minimal
    Cd = float(ck.get("Cd", (config['CD_L'] + config['CD_H']) / 2))
    Crr = float(ck.get("Crr", (config['CRR_L'] + config['CRR_H']) / 2))
    m = float(ck.get("m", (config['M_L'] + config['M_H']) / 2))
    Paux = float(ck.get("Paux", (config['PA_L'] + config['PA_H']) / 2))
    eta = float(ck.get("eta", (config['ETA_L'] + config['ETA_H']) / 2))
    etar = float(ck.get("etar", (config['ETAR_L'] + config['ETAR_H']) / 2))
    return Cd, Crr, m, Paux, eta, etar

def physics_power(v: np.ndarray, a: np.ndarray, vehicle: str, *, 
                 Cd: float, Crr: float, m: float, Paux: float, eta: float, etar: float) -> np.ndarray:
    """
    Vectorized physics power calculation (W).
    v [m/s], a [m/s^2] as 1D numpy arrays.
    """
    config = VEHICLE_CONFIGS[vehicle]
    
    aero = 0.5 * config['RHO'] * config['A'] * Cd * v**3
    roll = Crr * m * config['G'] * v
    iner = m * a * v
    Pm = aero + roll + iner
    
    Ptr = np.maximum(Pm, 0.0)  # traction (+)
    Prg = np.minimum(Pm, 0.0)  # regen (−)
    
    return Ptr / eta + etar * Prg + Paux  # note Prg is negative

def main():
    ap = argparse.ArgumentParser(f"Unified physics-only evaluator for all vehicles")
    ap.add_argument("vehicle", choices=list(VEHICLE_CONFIGS.keys()), 
                    help="Vehicle type to evaluate")
    ap.add_argument("--ckpt", help="Training checkpoint to read scalars (uses vehicle default if not specified)")
    ap.add_argument("--csv", help="CSV with Speed_SG [m/s], Acceleration_SG [m/s^2], BatteryPower_SG [W] (uses vehicle default test CSV if not specified)")
    ap.add_argument("--fs", type=float, default=10.0, help="Sampling rate (Hz) for PSD")
    ap.add_argument("--out_dir", help="Directory to save outputs (uses eval_out_{vehicle} if not specified)")
    ap.add_argument("--out_prefix", help="Filename prefix (uses vehicle name if not specified)")
    
    # Optional manual overrides (if provided, these take precedence)
    ap.add_argument("--Cd", type=float, help="Override drag coefficient")
    ap.add_argument("--Crr", type=float, help="Override rolling resistance")
    ap.add_argument("--m", type=float, help="Override mass")
    ap.add_argument("--Paux", type=float, help="Override auxiliary power")
    ap.add_argument("--eta", type=float, help="Override motor efficiency")
    ap.add_argument("--etar", type=float, help="Override regen efficiency")
    ap.add_argument("--save_fig", action="store_true", help="Save plots as files")
    ap.add_argument("--no_plot", action="store_true", help="Skip plotting")
    
    args = ap.parse_args()
    
    config = VEHICLE_CONFIGS[args.vehicle]
    
    # Set defaults
    ckpt_path = args.ckpt or f"models/{config['model_name']}"
    csv_path = args.csv or args.vehicle + "_test.csv"
    out_dir = Path(args.out_dir or f"eval_out_{args.vehicle}")
    out_prefix = args.out_prefix or f"{args.vehicle}_physics"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating {args.vehicle} model:")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  CSV: {csv_path}")
    print(f"  Output directory: {out_dir}")
    
    # Load or override scalars
    Cd0, Crr0, m0, Paux0, eta0, etar0 = load_scalars_from_ckpt(Path(ckpt_path), args.vehicle)
    Cd = args.Cd if args.Cd is not None else Cd0
    Crr = args.Crr if args.Crr is not None else Crr0
    m = args.m if args.m is not None else m0
    Paux = args.Paux if args.Paux is not None else Paux0
    eta = args.eta if args.eta is not None else eta0
    etar = args.etar if args.etar is not None else etar0

    print(f"\\nUsing physics parameters:")
    print(f"  Cd={Cd:.4f}, Crr={Crr:.5f}, m={m:.0f} kg")
    print(f"  Paux={Paux:.0f} W, η={eta:.3f}, ηreg={etar:.3f}")
    print()

    # Load data
    df = pd.read_csv(csv_path)
    v = df["Speed_SG"].values
    a = df["Acceleration_SG"].values
    P_meas = df["BatteryPower_SG"].values

    # Physics prediction
    P_pred = physics_power(v, a, args.vehicle, Cd=Cd, Crr=Crr, m=m, Paux=Paux, eta=eta, etar=etar)

    # Metrics
    rmse = sqrt(mean_squared_error(P_meas, P_pred))
    mae = mean_absolute_error(P_meas, P_pred)
    r2 = r2_score(P_meas, P_pred)

    print(f"Model evaluation results:")
    print(f"  RMSE: {rmse:.1f} W")
    print(f"  MAE:  {mae:.1f} W")
    print(f"  R²:   {r2:.4f}")
    print()

    # Derivative MAE (if sufficient data)
    if len(P_meas) > 10:
        dP_meas_dt = np.gradient(P_meas)
        dP_pred_dt = np.gradient(P_pred)
        d_mae = mean_absolute_error(dP_meas_dt, dP_pred_dt)
        print(f"  Derivative MAE: {d_mae:.1f} W/sample")
    else:
        d_mae = None

    # Save metrics
    metrics = {
        "vehicle": [args.vehicle],
        "rmse_W": [rmse],
        "mae_W": [mae], 
        "r2": [r2],
        "derivative_mae_W_per_sample": [d_mae],
        "Cd": [Cd],
        "Crr": [Crr],
        "m_kg": [m],
        "Paux_W": [Paux],
        "eta": [eta],
        "etar": [etar]
    }
    pd.DataFrame(metrics).to_csv(out_dir / f"{out_prefix}_metrics.csv", index=False)

    # Time-series comparison
    time_data = {
        "time": np.arange(len(P_meas)),
        "P_measured_W": P_meas,
        "P_predicted_W": P_pred,
        "residual_W": P_meas - P_pred,
        "speed_ms": v,
        "accel_ms2": a
    }
    pd.DataFrame(time_data).to_csv(out_dir / f"{out_prefix}_timeseries.csv", index=False)

    # Power spectral density
    if len(P_meas) >= 256:  # Need sufficient data for PSD
        f_meas, psd_meas = scipy.signal.welch(P_meas, fs=args.fs, nperseg=min(256, len(P_meas)//4))
        f_pred, psd_pred = scipy.signal.welch(P_pred, fs=args.fs, nperseg=min(256, len(P_pred)//4))
        
        psd_data = {
            "frequency_Hz": f_meas,
            "psd_measured": psd_meas,
            "psd_predicted": psd_pred
        }
        pd.DataFrame(psd_data).to_csv(out_dir / f"{out_prefix}_psd.csv", index=False)
    
    print(f"Results saved to {out_dir}/")
    print(f"  - {out_prefix}_metrics.csv")
    print(f"  - {out_prefix}_timeseries.csv")
    if len(P_meas) >= 256:
        print(f"  - {out_prefix}_psd.csv")

    # Plotting
    if not args.no_plot:
        print("\\nGenerating plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{args.vehicle.upper()} Model Evaluation", fontsize=16)
        
        # Time series
        axes[0, 0].plot(P_meas / 1e3, label='Measured', alpha=0.7)
        axes[0, 0].plot(P_pred / 1e3, label='Predicted', alpha=0.7)
        axes[0, 0].set_title('Power vs Time')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Power (kW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Correlation plot
        axes[0, 1].scatter(P_meas/1e3, P_pred/1e3, alpha=0.5, s=1)
        lim = [min(P_meas.min(), P_pred.min())/1e3, max(P_meas.max(), P_pred.max())/1e3]
        axes[0, 1].plot(lim, lim, 'r--', alpha=0.8)
        axes[0, 1].set_xlim(lim)
        axes[0, 1].set_ylim(lim)
        axes[0, 1].set_xlabel('Measured Power (kW)')
        axes[0, 1].set_ylabel('Predicted Power (kW)')
        axes[0, 1].set_title(f'Correlation (R² = {r2:.4f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals
        residuals = P_meas - P_pred
        axes[1, 0].plot(residuals/1e3, alpha=0.7)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 0].set_title(f'Residuals (MAE = {mae:.1f} W)')
        axes[1, 0].set_xlabel('Sample')
        axes[1, 0].set_ylabel('Residual (kW)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Power spectral density
        if len(P_meas) >= 256:
            axes[1, 1].loglog(f_meas, psd_meas/1e3, label='Measured', alpha=0.7)
            axes[1, 1].loglog(f_pred, psd_pred/1e3, label='Predicted', alpha=0.7)
            axes[1, 1].set_title('Power Spectral Density')
            axes[1, 1].set_xlabel('Frequency (Hz)')
            axes[1, 1].set_ylabel('PSD (kW²/Hz)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\\nfor PSD analysis', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Power Spectral Density')
        
        plt.tight_layout()
        
        if args.save_fig:
            fig_path = out_dir / f"{out_prefix}_evaluation.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"  - {fig_path}")
            plt.close()
        else:
            plt.show()

    print(f"\\n {args.vehicle.upper()} evaluation completed")

if __name__ == "__main__":
    main()