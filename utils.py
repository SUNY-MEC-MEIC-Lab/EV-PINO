#!/usr/bin/env python
"""
Common utilities for PINO training and evaluation.
Contains shared model architectures, data processing, and training functions.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

torch.set_default_dtype(torch.float32)

# ===============================================================================
#                    VEHICLE CONFIGURATIONS
# ===============================================================================

VEHICLE_CONFIGS = {
    'ev9': {
        'RHO': 1.17, 'A': 3.05, 'G': 9.81,
        'CD_L': 0.27, 'CD_H': 0.29,
        'CRR_L': 0.0055, 'CRR_H': 0.0075,
        'M_L': 2650., 'M_H': 2880.,
        'PA_L': 300., 'PA_H': 800.,
        'ETA_L': 0.60, 'ETA_H': 0.90,
        'ETAR_L': 0.20, 'ETAR_H': 0.80,
        'SPAN_ETA': 0.08, 'SPAN_ETAR': 0.25, 'SPAN_PAUX': 300.0,
        'HEAD_TEMP': 1.5, 'V0': 18.0, 'SV': 6.0,
        'eff_head_out': 3,  # d_eta, d_etar, d_paux
        'variable_paux': True,
        'default_csv': 'ev9_train.csv',
        'model_name': 'pino_ev9_params.pt',
        'log_name': 'training_ev9_log.csv'
    },
    'tesla3': {
        'RHO': 1.17, 'A': 2.22, 'G': 9.81,
        'CD_L': 0.23, 'CD_H': 0.25,
        'CRR_L': 0.009, 'CRR_H': 0.011,
        'M_L': 1950., 'M_H': 2000.,
        'PA_L': 1000., 'PA_H': 1200.,
        'ETA_L': 0.60, 'ETA_H': 0.90,
        'ETAR_L': 0.20, 'ETAR_H': 0.90,
        'SPAN_ETA': 0.08, 'SPAN_ETAR': 0.25, 'SPAN_PAUX': 0.0,
        'HEAD_TEMP': 1.5, 'V0': 18.0, 'SV': 6.0,
        'eff_head_out': 2,  # d_eta, d_etar only
        'variable_paux': False,
        'default_csv': 'tesla3_train.csv',
        'model_name': 'pino_tesla3_params.pt',
        'log_name': 'training_log_tesla3.csv'
    },
    'teslaS': {
        'RHO': 1.17, 'A': 2.40, 'G': 9.81,
        'CD_L': 0.235, 'CD_H': 0.25,
        'CRR_L': 0.010, 'CRR_H': 0.012,
        'M_L': 2200., 'M_H': 2400.,
        'PA_L': 0., 'PA_H': 1000.,
        'ETA_L': 0.60, 'ETA_H': 0.90,
        'ETAR_L': 0.20, 'ETAR_H': 0.90,
        'SPAN_ETA': 0.08, 'SPAN_ETAR': 0.25, 'SPAN_PAUX': 0.0,
        'HEAD_TEMP': 1.5, 'V0': 18.0, 'SV': 6.0,
        'eff_head_out': 2,  # d_eta, d_etar only
        'variable_paux': False,
        'default_csv': 'teslaS_train.csv',
        'model_name': 'pino_teslaS_params.pt',
        'log_name': 'training_log_teslaS.csv'
    }
}

# Default regularization parameters
DEFAULT_LAMBDA_CORR = 1e-5
DEFAULT_LAMBDA_SMOOTH = 5e-5
DEFAULT_LAMBDA_L2 = 5e-6
DEFAULT_LAMBDA_PAUX_SMOOTH_SCALE = 1e-6
DEFAULT_LAMBDA_PAUX_CENTER = 1e-8

# ===============================================================================
#                        UTILITY FUNCTIONS
# ===============================================================================

def sigmoid_bound(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Map raw parameter to bounded range using sigmoid."""
    return lo + (hi - lo) * torch.sigmoid(raw)

def windows(n: int, L: int, s: int) -> np.ndarray:
    """Generate sliding window indices."""
    return np.arange(0, n - L + 1, s)

def _unnorm(xb: torch.Tensor, scaler: StandardScaler, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Unnormalize input sequences to get velocity and acceleration."""
    mean = torch.as_tensor(scaler.mean_, dtype=torch.float32, device=device).view(1, 1, 2)
    std = torch.as_tensor(scaler.scale_, dtype=torch.float32, device=device).view(1, 1, 2)
    va = xb * std + mean
    return va[..., 0], va[..., 1]  # v [m/s], a [m/s^2]

# ===============================================================================
#                        DATASET CLASSES
# ===============================================================================

class SeqDS(Dataset):
    """Sequence dataset for PINO training."""
    def __init__(self, csv: str, idx: np.ndarray, scaler: StandardScaler, L: int):
        df = pd.read_csv(csv)
        x = scaler.transform(df[["Speed_SG", "Acceleration_SG"]].astype(np.float32))
        P = df["BatteryPower_SG"].astype(np.float32).values
        self.x = np.stack([x[s:s+L] for s in idx])
        self.Pm = np.stack([P[s:s+L] for s in idx])
        
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.Pm[i])

def prepare_data(csv: str, *, L: int = 128, stride: int = 32, batch: int = 128, 
                val: float = 0.10, test: float = 0.02) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """Prepare training, validation, and test data loaders."""
    df = pd.read_csv(csv)
    idx = windows(len(df), L, stride)
    sx = StandardScaler().fit(df[["Speed_SG", "Acceleration_SG"]])
    
    i_tr, i_tmp = train_test_split(idx, test_size=val+test, random_state=42, shuffle=True)
    i_va, i_te = train_test_split(i_tmp, test_size=val/(val+test), random_state=42, shuffle=True)
    
    mk = lambda ids, sh: DataLoader(SeqDS(csv, ids, sx, L), batch_size=batch, shuffle=sh, drop_last=sh)
    return mk(i_tr, True), mk(i_va, False), mk(i_te, False), sx

# ===============================================================================
#                      MODEL ARCHITECTURES
# ===============================================================================

class SpectralConv1d(nn.Module):
    """1D Spectral convolution layer for FNO."""
    def __init__(self, c_in: int, c_out: int, modes: int):
        super().__init__()
        self.m = modes
        scale = 1 / (c_in * c_out)
        self.wr = nn.Parameter(scale * torch.randn(c_in, c_out, modes))
        self.wi = nn.Parameter(scale * torch.randn(c_in, c_out, modes))
        self.pad = modes // 4
        
    @staticmethod
    def _cmul(x: torch.Tensor, wr: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in Fourier space."""
        xr, xi = x[..., 0], x[..., 1]
        re = (torch.einsum("bck,cok->bok", xr, wr) - torch.einsum("bck,cok->bok", xi, wi))
        im = (torch.einsum("bck,cok->bok", xr, wi) + torch.einsum("bck,cok->bok", xi, wr))
        return torch.stack([re, im], -1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        x = F.pad(x, (0, self.pad), "reflect")
        Np = x.shape[-1]
        X = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm="forward"))
        out = torch.zeros(B, self.wr.size(1), X.size(-2), 2, device=x.device, dtype=x.dtype)
        k = min(self.m, X.size(-2))
        out[:, :, :k] = self._cmul(X[:, :, :k], self.wr[:, :, :k], self.wi[:, :, :k])
        x = torch.fft.irfft(torch.view_as_complex(out), n=Np, dim=-1, norm="forward")
        return x[..., :N]

class FNO1d(nn.Module):
    """1D Fourier Neural Operator trunk."""
    def __init__(self, *, modes: int = 16, width: int = 128, layers: int = 4):
        super().__init__()
        self.lift = nn.Sequential(nn.Linear(3, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.spec = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(),
                                               nn.Conv1d(width, width, 1)) for _ in range(layers)])
        self.act = nn.GELU()
    
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input sequences. x: (B,L,2)"""
        B, L, _ = x.shape
        grid = torch.linspace(0, 1, L, device=x.device).view(1, L, 1).expand(B, -1, -1)
        x = self.lift(torch.cat([x, grid], -1)).permute(0, 2, 1)  # (B,W,L)
        for sc, mlp in zip(self.spec, self.mlp):
            x = x + self.act(sc(x) + mlp(x))
        return x  # (B,W,L)

class PINO(nn.Module):
    """Physics-Informed Neural Operator with vehicle-specific configuration."""
    def __init__(self, config: dict, *, modes: int = 16, width: int = 128, layers: int = 4):
        super().__init__()
        self.config = config
        self.trunk = FNO1d(modes=modes, width=width, layers=layers)
        self.eff_head = nn.Conv1d(width, config['eff_head_out'], 1)
        self.res_head = nn.Conv1d(width, 1, 1)
        
        nn.init.zeros_(self.res_head.weight)
        nn.init.zeros_(self.res_head.bias)
        
        # Constant scalar baselines
        self.raw_cd = nn.Parameter(torch.tensor(0.0))
        self.raw_crr = nn.Parameter(torch.tensor(0.0))
        self.raw_m = nn.Parameter(torch.tensor(0.0))
        self.raw_paux = nn.Parameter(torch.tensor(0.0))
        self.raw_eta = nn.Parameter(torch.tensor(0.0))
        self.raw_etar = nn.Parameter(torch.tensor(0.0))
    
    def baselines(self) -> tuple[torch.Tensor, ...]:
        """Get bounded scalar parameters."""
        cfg = self.config
        Cd = sigmoid_bound(self.raw_cd, cfg['CD_L'], cfg['CD_H'])
        Crr = sigmoid_bound(self.raw_crr, cfg['CRR_L'], cfg['CRR_H'])
        m = sigmoid_bound(self.raw_m, cfg['M_L'], cfg['M_H'])
        Paux = sigmoid_bound(self.raw_paux, cfg['PA_L'], cfg['PA_H'])
        eta0 = sigmoid_bound(self.raw_eta, cfg['ETA_L'], cfg['ETA_H'])
        er0 = sigmoid_bound(self.raw_etar, cfg['ETAR_L'], cfg['ETAR_H'])
        return Cd, Crr, m, Paux, eta0, er0
    
    def eff_time(self, feat: torch.Tensor, v: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute time-varying efficiency parameters. feat: (B,W,L), v: (B,L)"""
        cfg = self.config
        Cd0, Crr0, m0, Paux0, eta0, er0 = self.baselines()
        
        # Raw deltas from head
        d = torch.tanh(self.eff_head(feat) / cfg['HEAD_TEMP'])  # (B,out_channels,L)
        
        # Speed gate for efficiencies
        w_hi = torch.sigmoid((v - cfg['V0']) / cfg['SV']).clamp(0, 1)
        
        # Time-varying parameters
        eta_t = torch.clamp(eta0.view(-1, 1) + cfg['SPAN_ETA'] * (w_hi * d[:, 0]), 
                           cfg['ETA_L'], cfg['ETA_H'])
        er_t = torch.clamp(er0.view(-1, 1) + cfg['SPAN_ETAR'] * (w_hi * d[:, 1]), 
                          cfg['ETAR_L'], cfg['ETAR_H'])
        
        if cfg['variable_paux']:
            # EV9: variable Paux
            w_ax = torch.ones_like(w_hi)
            Paux_t = torch.clamp(Paux0.view(-1, 1) + cfg['SPAN_PAUX'] * (w_ax * d[:, 2]),
                               cfg['PA_L'], cfg['PA_H'])
        else:
            # Tesla: constant Paux
            Paux_t = Paux0.view(-1, 1).expand_as(eta_t)
        
        return dict(Cd=Cd0, Crr=Crr0, m=m0, Paux=Paux_t, eta=eta_t, er=er_t)
    
    def physics(self, v: torch.Tensor, a: torch.Tensor, vt: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute physics-based power consumption."""
        cfg = self.config
        Cd, Crr, m, Paux, eta, er = vt["Cd"], vt["Crr"], vt["m"], vt["Paux"], vt["eta"], vt["er"]
        
        aero = 0.5 * cfg['RHO'] * cfg['A'] * Cd * v**3
        roll = Crr * m * cfg['G'] * v
        iner = m * a * v
        Pm = aero + roll + iner
        
        Ptr = torch.relu(Pm)   # Traction (positive)
        Prg = -torch.relu(-Pm) # Regeneration (negative)
        
        return Ptr / eta + Prg * er + Paux
    
    def forward(self, xb: torch.Tensor, v: torch.Tensor, a: torch.Tensor, *, 
               use_eff: bool = True, return_parts: bool = False):
        """Forward pass through PINO model."""
        if use_eff:
            feat = self.trunk.extract(xb)  # (B,W,L)
            vt = self.eff_time(feat, v)    # dict with time-varying parameters
            base = self.physics(v, a, vt)  # (B,L)
            corr = self.res_head(feat).squeeze(1)  # (B,L)
            tot = base + corr
            if return_parts:
                return tot, base, corr, vt
            return tot
        else:
            # Physics with constant baselines (no heads)
            cfg = self.config
            Cd, Crr, m, Paux, eta0, er0 = self.baselines()
            aero = 0.5 * cfg['RHO'] * cfg['A'] * Cd * v**3
            roll = Crr * m * cfg['G'] * v
            iner = m * a * v
            Pm = aero + roll + iner
            Ptr = torch.relu(Pm)
            Prg = -torch.relu(-Pm)
            return Ptr / eta0 + Prg * er0 + Paux

# ===============================================================================
#                    TRAINING HELPER FUNCTIONS
# ===============================================================================

@torch.no_grad()
def collect_val_stats(model: PINO, loader: DataLoader, scaler: StandardScaler, 
                     device: torch.device, *, use_eff: bool = True) -> dict[str, float]:
    """Compute validation metrics and parameter percentiles for logging."""
    model.eval()
    all_true, all_pred = [], []
    eta_vals, er_vals, pa_vals = [], [], []
    
    for xb, Pm in loader:
        xb, Pm = xb.to(device), Pm.to(device)
        v, a = _unnorm(xb, scaler, device)
        
        if use_eff:
            pred_t, _, _, vt = model(xb, v, a, use_eff=True, return_parts=True)
            eta_vals.append(vt["eta"].detach().cpu().numpy().ravel())
            er_vals.append(vt["er"].detach().cpu().numpy().ravel())
            if model.config['variable_paux']:
                pa_vals.append(vt["Paux"].detach().cpu().numpy().ravel())
        else:
            pred_t = model(xb, v, a, use_eff=False)
            
        all_true.append(Pm.detach().cpu().numpy().ravel())
        all_pred.append(pred_t.detach().cpu().numpy().ravel())
    
    y = np.concatenate(all_true)
    yhat = np.concatenate(all_pred)
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    mae = float(mean_absolute_error(y, yhat))
    r2 = float(r2_score(y, yhat))
    
    stats = dict(val_rmse_W=rmse, val_mae_W=mae, val_r2=r2)
    
    if use_eff and eta_vals:
        eta_all = np.concatenate(eta_vals)
        er_all = np.concatenate(er_vals)
        
        def pct(arr: np.ndarray, q: float) -> float:
            return float(np.percentile(arr, q))
        
        stats.update({
            "eta_min": float(np.min(eta_all)), "eta_max": float(np.max(eta_all)),
            "eta_p05": pct(eta_all, 5), "eta_p95": pct(eta_all, 95),
            "er_min": float(np.min(er_all)), "er_max": float(np.max(er_all)),
            "er_p05": pct(er_all, 5), "er_p95": pct(er_all, 95),
        })
        
        if model.config['variable_paux'] and pa_vals:
            pa_all = np.concatenate(pa_vals)
            stats.update({
                "paux_min": float(np.min(pa_all)), "paux_max": float(np.max(pa_all)),
                "paux_p05": pct(pa_all, 5), "paux_p95": pct(pa_all, 95),
            })
    
    return stats

def run_epoch(model: PINO, loader: DataLoader, scaler: StandardScaler, device: torch.device, 
             *, opt: torch.optim.Optimizer = None, use_eff: bool = True, 
             lam_corr: float = DEFAULT_LAMBDA_CORR, lam_smooth: float = DEFAULT_LAMBDA_SMOOTH, 
             lam_l2: float = DEFAULT_LAMBDA_L2, 
             lam_paux_smooth_scale: float = DEFAULT_LAMBDA_PAUX_SMOOTH_SCALE,
             lam_paux_center: float = DEFAULT_LAMBDA_PAUX_CENTER) -> float:
    """Run one training/validation epoch."""
    train = opt is not None
    model.train() if train else model.eval()
    tot, n = 0.0, 0
    
    for xb, Pm in loader:
        xb, Pm = xb.to(device), Pm.to(device)
        v, a = _unnorm(xb, scaler, device)
        
        if train:
            opt.zero_grad()
        
        if use_eff:
            pred, base, corr, vt = model(xb, v, a, use_eff=True, return_parts=True)
            loss = F.mse_loss(pred / 1000., Pm / 1000.)  # kW^2
            loss = loss + lam_corr * (corr / 1000.).pow(2).mean()  # keep residual modest
            
            # Smoothness on time-varying parameters
            d_eta = vt["eta"][:, 1:] - vt["eta"][:, :-1]
            d_er = vt["er"][:, 1:] - vt["er"][:, :-1]
            loss = loss + lam_smooth * (d_eta.pow(2).mean() + d_er.pow(2).mean())
            
            if model.config['variable_paux']:
                d_pa = vt["Paux"][:, 1:] - vt["Paux"][:, :-1]
                loss = loss + (lam_smooth * lam_paux_smooth_scale) * d_pa.pow(2).mean()
                
                # Optional tiny centering around baseline scalar Paux
                Paux0 = model.baselines()[3].view(-1, 1)
                loss = loss + lam_paux_center * (vt["Paux"] - Paux0).pow(2).mean()
        else:
            pred = model(xb, v, a, use_eff=False)
            loss = F.mse_loss(pred / 1000., Pm / 1000.)
        
        # Small L2 on raw scalars
        raw_params = [model.raw_cd, model.raw_crr, model.raw_m, 
                     model.raw_paux, model.raw_eta, model.raw_etar]
        loss = loss + lam_l2 * sum(p.pow(2) for p in raw_params)
        
        if train:
            loss.backward()
            opt.step()
            
        bs = xb.size(0)
        tot += loss.item() * bs
        n += bs
    
    return tot / n

# ===============================================================================
#                      MAIN TRAINING FUNCTION
# ===============================================================================

def train_vehicle(vehicle: str, csv: str = None, *, L: int = 128, stride: int = 32, batch: int = 128,
                 modes: int = 16, width: int = 128, layers: int = 4,
                 epochs: int = 3000, lr: float = 3e-4, patience: int = 200, warmup: int = 400,
                 lam_corr: float = DEFAULT_LAMBDA_CORR, lam_smooth: float = DEFAULT_LAMBDA_SMOOTH, 
                 lam_l2: float = DEFAULT_LAMBDA_L2, 
                 lam_paux_smooth_scale: float = DEFAULT_LAMBDA_PAUX_SMOOTH_SCALE,
                 lam_paux_center: float = DEFAULT_LAMBDA_PAUX_CENTER,
                 log_csv: str = None, model_save_path: str = None) -> None:
    """Unified training function for all vehicle types."""
    
    if vehicle not in VEHICLE_CONFIGS:
        raise ValueError(f"Unknown vehicle type: {vehicle}. Available: {list(VEHICLE_CONFIGS.keys())}")
    
    config = VEHICLE_CONFIGS[vehicle]
    
    # Use defaults if not provided
    if csv is None:
        csv = config['default_csv']
    if log_csv is None:
        log_csv = config['log_name']
    if model_save_path is None:
        model_save_path = config['model_name']
    
    print(f"Training {vehicle} model on {csv}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te, sx = prepare_data(csv, L=L, stride=stride, batch=batch)
    model = PINO(config, modes=modes, width=width, layers=layers).to(device)
    
    logs = []
    t0 = time.time()
    
    # Warmup Phase
    print(f"Starting warmup phase ({warmup} epochs)...")
    for p in model.trunk.parameters():
        p.requires_grad_(False)
    for p in model.eff_head.parameters():
        p.requires_grad_(False)
    for p in model.res_head.parameters():
        p.requires_grad_(False)
        
    opt_w = torch.optim.Adam([model.raw_cd, model.raw_crr, model.raw_m,
                             model.raw_paux, model.raw_eta, model.raw_etar], lr=lr)
    
    for ep in range(1, warmup + 1):
        tr_l = run_epoch(model, tr, sx, device, opt=opt_w, use_eff=False,
                        lam_corr=lam_corr, lam_smooth=lam_smooth, lam_l2=lam_l2,
                        lam_paux_smooth_scale=lam_paux_smooth_scale, lam_paux_center=lam_paux_center)
        va_l = run_epoch(model, va, sx, device, use_eff=False,
                        lam_corr=lam_corr, lam_smooth=lam_smooth, lam_l2=lam_l2,
                        lam_paux_smooth_scale=lam_paux_smooth_scale, lam_paux_center=lam_paux_center)
        
        Cd, Crr, m, Paux, eta0, er0 = model.baselines()
        lr_now = opt_w.param_groups[0]["lr"]
        
        # Validation stats (physics-only)
        vstats = collect_val_stats(model, va, sx, device, use_eff=False)
        
        logs.append({
            "phase": "warmup", "epoch": ep,
            "train_mse_kW2": tr_l, "val_mse_kW2": va_l,
            "val_rmse_W": vstats["val_rmse_W"], "val_mae_W": vstats["val_mae_W"], "val_r2": vstats["val_r2"],
            "lr": lr_now,
            "Cd0": float(Cd), "Crr0": float(Crr), "m0": float(m), "Paux0": float(Paux),
            "eta0": float(eta0), "eta_reg0": float(er0),
            "elapsed_s": time.time() - t0
        })
        
        if ep == 1 or ep % 25 == 0:
            print(f"[warmup] ep {ep:4d} | train {tr_l:6.2f} kW² | val {va_l:6.2f} kW² | "
                  f"Cd0={Cd:.3f} Crr0={Crr:.4f} m={m:.0f} Paux0={Paux:.0f} η0≈{eta0:.3f} ηreg0≈{er0:.3f}")
        
        pd.DataFrame(logs).to_csv(log_csv, index=False)
    
    # Full Training Phase
    print(f"Starting full training phase ({epochs} epochs)...")
    for p in model.trunk.parameters():
        p.requires_grad_(True)
    for p in model.eff_head.parameters():
        p.requires_grad_(True)
    for p in model.res_head.parameters():
        p.requires_grad_(True)
        
    opt = torch.optim.Adam([
        {"params": model.trunk.parameters()},
        {"params": model.eff_head.parameters()},
        {"params": model.res_head.parameters()},
        {"params": [model.raw_cd, model.raw_crr, model.raw_m,
                   model.raw_paux, model.raw_eta, model.raw_etar]}
    ], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    
    best, best_state, no_imp = np.inf, None, 0
    
    for ep in range(1, epochs + 1):
        tr_l = run_epoch(model, tr, sx, device, opt=opt, use_eff=True,
                        lam_corr=lam_corr, lam_smooth=lam_smooth, lam_l2=lam_l2,
                        lam_paux_smooth_scale=lam_paux_smooth_scale, lam_paux_center=lam_paux_center)
        va_l = run_epoch(model, va, sx, device, use_eff=True,
                        lam_corr=lam_corr, lam_smooth=lam_smooth, lam_l2=lam_l2,
                        lam_paux_smooth_scale=lam_paux_smooth_scale, lam_paux_center=lam_paux_center)
        
        if va_l < best:
            best, best_state, no_imp = va_l, model.state_dict(), 0
        else:
            no_imp += 1
            
        sch.step()
        
        Cd, Crr, m, Paux, eta0, er0 = model.baselines()
        lr_now = sch.get_last_lr()[0]
        
        # Validation stats with variable heads
        vstats = collect_val_stats(model, va, sx, device, use_eff=True)
        
        row = {
            "phase": "train", "epoch": ep,
            "train_mse_kW2": tr_l, "val_mse_kW2": va_l,
            "val_rmse_W": vstats["val_rmse_W"], "val_mae_W": vstats["val_mae_W"], "val_r2": vstats["val_r2"],
            "lr": lr_now,
            "Cd0": float(Cd), "Crr0": float(Crr), "m0": float(m), "Paux0": float(Paux),
            "eta0": float(eta0), "eta_reg0": float(er0),
            "elapsed_s": time.time() - t0
        }
        
        # Add parameter percentiles
        for k in ("eta_min", "eta_max", "eta_p05", "eta_p95", "er_min", "er_max", "er_p05", "er_p95",
                 "paux_min", "paux_max", "paux_p05", "paux_p95"):
            if k in vstats:
                row[k] = vstats[k]
                
        logs.append(row)
        
        if ep == 1 or ep % 25 == 0:
            print(f"ep {ep:4d} | train {tr_l:6.2f} kW² | val {va_l:6.2f} kW² | "
                  f"Cd0={Cd:.3f} Crr0={Crr:.4f} m={m:.0f} Paux0={Paux:.0f} η0≈{eta0:.3f} ηreg0≈{er0:.3f}")
            
        if no_imp >= patience:
            print("Early stopping")
            break
            
        pd.DataFrame(logs).to_csv(log_csv, index=False)
    
    # Final Evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
        
    test_mse = run_epoch(model, te, sx, device, use_eff=True,
                        lam_corr=lam_corr, lam_smooth=lam_smooth, lam_l2=lam_l2,
                        lam_paux_smooth_scale=lam_paux_smooth_scale, lam_paux_center=lam_paux_center)
    vstats_te = collect_val_stats(model, te, sx, device, use_eff=True)
    
    print(f"\n· Best val MSE {best:.2f} kW² · Test MSE {test_mse:.2f} kW²")
    print(f"· Test RMSE {vstats_te['val_rmse_W']:.0f} W | MAE {vstats_te['val_mae_W']:.0f} W | R² {vstats_te['val_r2']:.4f}")
    
    # Final save
    torch.save({
        "state_dict": model.state_dict(),
        "scaler_x": sx,
        "seq_len": L,
        "modes": modes, "width": width, "layers": layers,
        "vehicle": vehicle,
        "config": config
    }, model_save_path)
    
    Cd, Crr, m, Paux, eta0, er0 = model.baselines()
    print(f"Saved → {model_save_path}")
    print(f"Final scalars — Cd0={Cd:.3f} Crr0={Crr:.4f} m={m:.0f} Paux0={Paux:.0f} η0≈{eta0:.3f} ηreg0≈{er0:.3f}")
    
    # Append final test metrics row & write CSV
    logs.append({
        "phase": "final_test", "epoch": None,
        "train_mse_kW2": None, "val_mse_kW2": best,
        "test_mse_kW2": test_mse,
        "test_rmse_W": vstats_te["val_rmse_W"],
        "test_mae_W": vstats_te["val_mae_W"],
        "test_r2": vstats_te["val_r2"],
        "lr": 0.0,
        "Cd0": float(Cd), "Crr0": float(Crr), "m0": float(m), "Paux0": float(Paux),
        "eta0": float(eta0), "eta_reg0": float(er0),
        "elapsed_s": time.time() - t0
    })
    pd.DataFrame(logs).to_csv(log_csv, index=False)
    print(f"Logged → {log_csv}")