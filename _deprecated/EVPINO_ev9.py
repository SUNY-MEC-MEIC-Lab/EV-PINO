#!/usr/bin/env python
# -----------------------------------------------------------------------------
#  EVPINO_ev9.py — Kia EV9 RWD training script (DEPRECATED)
#  Use EVPINO_train.py with 'ev9' argument instead:
#    python EVPINO_train.py ev9 [options]
# -----------------------------------------------------------------------------

import argparse
from utils import train_vehicle

def train(csv, **kwargs):
    """DEPRECATED: Use EVPINO_train.py ev9 instead."""
    print("WARNING: This script is deprecated. Use EVPINO_train.py instead:")
    print(f"  python EVPINO_train.py ev9 --csv {csv}")
    
    # Call new unified training function
    train_vehicle('ev9', csv=csv, **kwargs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser("DEPRECATED: Use EVPINO_train.py ev9 instead")
    ap.add_argument("--csv", default="ev9_train.csv", help="CSV file path")
    ap.add_argument("--epochs", type=int, default=3000, help="Training epochs")
    ap.add_argument("--L", type=int, default=32, help="Sequence length")
    ap.add_argument("--stride", type=int, default=2, help="Window stride")
    ap.add_argument("--batch", type=int, default=256, help="Batch size")
    ap.add_argument("--modes", type=int, default=4, help="Fourier modes")
    ap.add_argument("--width", type=int, default=128, help="Model width")
    ap.add_argument("--layers", type=int, default=4, help="FNO layers")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--warmup", type=int, default=600, help="Warmup epochs")
    ap.add_argument("--log_csv", default="training_ev9_log.csv", help="Log file")
    args = ap.parse_args()
    
    print("⚠️  DEPRECATED: This script is deprecated!")
    print("   Use the new unified training script instead:")
    print(f"   python EVPINO_train.py ev9 --csv {args.csv} --epochs {args.epochs} --L {args.L}")
    print()
    
    # Still work for backward compatibility
    train_vehicle(
        'ev9',
        csv=args.csv,
        L=args.L,
        stride=args.stride,
        batch=args.batch,
        modes=args.modes,
        width=args.width,
        layers=args.layers,
        epochs=args.epochs,
        lr=args.lr,
        warmup=args.warmup,
        log_csv=args.log_csv
    )
