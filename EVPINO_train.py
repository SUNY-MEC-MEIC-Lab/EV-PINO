import argparse
from utils import train_vehicle, VEHICLE_CONFIGS

def main():
    ap = argparse.ArgumentParser("Unified PINO training for all vehicle types")
    ap.add_argument("vehicle", choices=list(VEHICLE_CONFIGS.keys()), 
                    help="Vehicle type to train")
    ap.add_argument("--csv", help="CSV file path (uses vehicle default if not specified)")
    ap.add_argument("--epochs", type=int, default=3000, help="Training epochs")
    ap.add_argument("--L", type=int, default=32, help="Sequence length")
    ap.add_argument("--stride", type=int, default=2, help="Window stride")
    ap.add_argument("--batch", type=int, default=256, help="Batch size")
    ap.add_argument("--modes", type=int, default=4, help="Fourier modes")
    ap.add_argument("--width", type=int, default=128, help="Model width")
    ap.add_argument("--layers", type=int, default=4, help="FNO layers")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--warmup", type=int, default=400, help="Warmup epochs (EV9: 600, Tesla: 400)")
    ap.add_argument("--patience", type=int, default=200, help="Early stopping patience")
    ap.add_argument("--log_csv", help="Log CSV file (uses vehicle default if not specified)")
    ap.add_argument("--model_save_path", help="Model save path (uses vehicle default if not specified)")
    
    # Regularization parameters
    ap.add_argument("--lam_corr", type=float, default=1e-5, help="Residual correlation penalty")
    ap.add_argument("--lam_smooth", type=float, default=5e-5, help="Smoothness penalty")
    ap.add_argument("--lam_l2", type=float, default=5e-6, help="L2 penalty on raw parameters")
    ap.add_argument("--lam_paux_smooth_scale", type=float, default=1e-6, help="Paux smoothness scale")
    ap.add_argument("--lam_paux_center", type=float, default=1e-8, help="Paux centering penalty")
    
    args = ap.parse_args()
    
    # Adjust default warmup for different vehicles
    if args.vehicle == 'ev9' and args.warmup == 400:
        args.warmup = 600
    
    print(f"Training {args.vehicle} model with the following configuration:")
    print(f"  Vehicle: {args.vehicle}")
    print(f"  CSV: {args.csv or VEHICLE_CONFIGS[args.vehicle]['default_csv']}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Sequence length: {args.L}")
    print(f"  Batch size: {args.batch}")
    print(f"  Learning rate: {args.lr}")
    print()
    
    train_vehicle(
        vehicle=args.vehicle,
        csv=args.csv,
        L=args.L,
        stride=args.stride,
        batch=args.batch,
        modes=args.modes,
        width=args.width,
        layers=args.layers,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        warmup=args.warmup,
        lam_corr=args.lam_corr,
        lam_smooth=args.lam_smooth,
        lam_l2=args.lam_l2,
        lam_paux_smooth_scale=args.lam_paux_smooth_scale,
        lam_paux_center=args.lam_paux_center,
        log_csv=args.log_csv,
        model_save_path=args.model_save_path
    )

if __name__ == "__main__":
    main()
