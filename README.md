# Human Pose Forecasting with Transformers

This project predicts future human body poses using a Transformer model and VPoser latent space encoding.  
It is based on the AMASS dataset and SMPL body model.

## Project Structure

- `hyperparameter_tuning.py` — Main script for model training and hyperparameter optimization (Optuna)
- `requirements.txt` — Python dependencies
- `run_tuning.sh` — Bash script for reproducible training
- `dataset/` — (Ignored) Data files for training/validation/testing
- `VPoserModelFiles/` — (Ignored) Pretrained VPoser model files
- `tuning_results/` — (Ignored) Output models, logs, and plots

## Setup

1. **Clone the repo:**
https://github.com/venkataseshtej/human-pose-forecasting

cd human-pose-forecasting

2. **Create and activate a virtual environment:**
python3 -m venv pose_env
source pose_env/bin/activate

3. **Install requirements:**
pip install -r requirements.txt


4. **Download and place the AMASS dataset and VPoser model in the correct folders.**

## Usage

Run hyperparameter tuning:
nohup ./run_tuning.sh > script_output.log 2>&1 &

All logs and output will be saved in `tuning_results/`.

## Results

- Best hyperparameters and model checkpoints are saved in `tuning_results/`.
- Training and validation loss curves, Optuna study plots, and parameter importances are also saved.

## License

[MIT](LICENSE) 

## Acknowledgements

- [AMASS Dataset](https://amass.is.tue.mpg.de/)
- [SMPL Model](https://smpl.is.tue.mpg.de/)
- [VPoser](https://smpl-x.is.tue.mpg.de/)
