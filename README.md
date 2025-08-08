# EV-PINO: EV-PINO: A Hybrid Surrogate for Electric Vehicle Parameter and Power Estimation Using Physics-Informed Neural Operator
This repository contains the source code for the EV-PINO project, which implements a Physics-Informed Neural Operator (PINO) to accurately predict battery power consumption for Electric Vehicles (EVs). The models are trained and evaluated on real-world driving datasets from the Tesla Model 3, and Tesla Model S, and Kia EV9.

**Abstract**
This paper presents EV-PINO, a hybrid surrogate model that couples a deterministic Electric Vehicle power model with a Physics-Informed Neural Operator for EV dynamics estimation. The model maps vehicle speed and acceleration data to key dynamic parameters—including aerodynamic drag, rolling resistance, mass, motor efficiency, regenerative braking efficiency, and auxiliary power. These learned parameters then drive a deterministic physics plant to produce final battery-power predictions. Validated on real-world driving data from a Tesla Model 3, Model S, and Kia EV9, EV-PINO excels at accurately identifying parameters that reflect the vehicle’s true, current operating state. This capability is crucial as it accounts for real-world variations from component aging, changing loads, and auxiliary system use, which are not captured by static factory specs. The model captures global sequence context with FFT spectral layers for efficient training and inference. The EV-PINO framework is modular, robust against distribution shifts and noise, and highly interpretable. It provides a practical surrogate for EV energy prediction and monitoring.

## Core Concepts
EV-PINO uses Fourier Neural Operators as its backbone. In process of mapping velocity and acceleration data into Battery Power, it estimates key parameters for EV dynamics:

- **Aerodynamic drag forces** (C<sub>d</sub>)
- **Rolling resistance** (C<sub>rr</sub>)  
- **Mass** (m)
- **Motor efficiency** (η)
- **Regenerative braking efficiency**(μ)
- **Auxiliary power consumption** (P<sub>aux</sub>)


## Repository Structure

```
/
├─── EVPINO_train.py           # Unified training script for all vehicle types
├─── EVPINO_test.py            # Unified testing script for physics-only evaluation
├─── utils.py                  # Shared model architectures and training utilities
├─── split.py                  # Time-series data splitting utility
├─── setup.py                  # Package installation configuration
├─── requirements.txt          # Python package dependencies
├─── Tesla3data/               # Tesla Model 3 dataset directory
│    └─── Tesla3_full.csv      # Tesla Model 3 complete telemetry data
├─── test_sample_data/         # Sample test datasets
│    └─── tesla3_test.csv      # Tesla Model 3 test sample
├─── train_sample_data/        # Sample training datasets
│    └─── tesla3_train.csv     # Tesla Model 3 training sample
├─── models/                   # Pre-trained model parameters (.pt files)
│    ├─── pino_tesla3_params.pt
│    ├─── pino_teslaS_params.pt
│    └─── pino_ev9_params.pt
├─── traininglogs/             # Training history and logs
│    ├─── training_log_tesla3.csv
│    ├─── training_log_teslaS.csv
│    └─── training_ev9_log.csv
├─── eval_out_*/               # Evaluation outputs (metrics, plots, logs)
│    ├─── eval_out_tesla3/
│    ├─── eval_out_teslaS/
│    └─── eval_out_ev9/
└─── _deprecated/              # Legacy individual model scripts
     ├─── PINO_EV9_test.py
     ├─── PINO_tesla3_test.py
     └─── PINO_teslaS_test.py
```

## Installation

This project requires **Python 3.10.16**. It is strongly recommended to use a virtual environment to manage dependencies.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd EVPINO
    ```

2.  **Install dependencies:**
    The required packages are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, for development, you can install the project in editable mode using the `setup.py` file:
    ```bash
    pip install -e .
    ```

## Usage

The primary entry points for using the models are the training and evaluation scripts.

### Model Training

To train a model for a specific vehicle, use the `EVPINO_train.py` script. Specify the vehicle type as a positional argument. You can also provide a path to a specific training data CSV file.

```bash
# Train with default settings for a specific vehicle
python EVPINO_train.py [ev9|tesla3|teslaS] --csv [path/to/csv_file]

# Example with custom arguments
python EVPINO_train.py ev9 --csv ev9_SG.csv --epochs 3000 --lr 3e-4 --batch 256
```
*   `vehicle`: The target vehicle type (positional argument).
*   `--csv`: (Optional) Path to the training CSV file.
*   `--epochs`: Number of training epochs.
*   `--lr`: Learning rate.

Trained model parameters will be saved in the `/models` directory.

### Model Evaluation

To evaluate a trained model, use the `EVPINO_test.py` script. This will load the corresponding pre-trained model and test data, run inference, and save the results (metrics, plots) to an `eval_out_*` directory.

```bash
# Evaluate with the default test set for a specific vehicle
python EVPINO_test.py [ev9|tesla3|teslaS] --csv [path/to/csv_file]

# Example with a specific CSV file
python EVPINO_test.py tesla3 --csv test_sample_data/tesla3_test.csv
```
Specific evaluation scripts are also available for each model (e.g., `PINO_EV9_test.py`).

## Data Availability

**Tesla Model 3 Data:** Full training and testing data is available for Tesla Model 3 (`Tesla3_JW_SG.csv`).

**Tesla Model S and Kia EV9 Training Data:** The training and testing datasets for Tesla Model S and Kia EV9 models cannot be publicly released as we have not yet received open sourcing permission from our partner organizations. However, pre-trained model weights (`.pt` files) and training logs are available in the repository for reference and evaluation purposes.

For researchers interested in the Tesla S or EV9 models, please refer to the provided model weights and training logs.

## Citation

If you use this code or the accompanying research in your work, please cite the original paper.
