# hyperparameter_tuning.py
import os
import sys
import logging
import pickle
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import optuna
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from torch.cuda.amp import autocast, GradScaler
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# Configure settings
DATA_ROOT = '/scratch/vmm5481/CV/dataset'  # Update with your lab server path
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'vali')
TEST_DIR = os.path.join(DATA_ROOT, 'test')
VPOSER_DIR = '/scratch/vmm5481/CV/VPoserModelFiles/vposer_v2_05'  # Update with your lab server path
OUTPUT_DIR = './tuning_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'tuning.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set random seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Check for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Positional Encoding (Batch-First Compatible)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:seq_len].unsqueeze(0)  # (1, seq_len, d_model)
        return x + pe.to(x.device)

# Transformer Model for Latent Forecasting
class MinimalPoseTransformer(nn.Module):
    def __init__(self, input_dim=32, latent_dim=32, d_model=256,
                 nhead=8, num_layers=6, dropout=0.1, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.input_proj = nn.Linear(input_dim, d_model)  # 32 -> d_model
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, latent_dim)  # d_model//2 -> 32
        )

    def forward(self, x):
        if self.verbose:
            logger.debug(f"Input shape: {x.shape}")

        x = self.input_proj(x)
        if self.verbose:
            logger.debug(f"After input_proj: {x.shape}")

        x = self.pos_encoder(x)
        if self.verbose:
            logger.debug(f"After PositionalEncoding: {x.shape}")

        x = self.encoder(x)
        if self.verbose:
            logger.debug(f"Encoded representation: {x.shape}")

        out = self.output_proj(x[:, -1])  # use last token
        if self.verbose:
            logger.debug(f"Output (predicted latent): {out.shape}")

        return out

def load_amass_pose_latents(folder_path, vp_model, device):
    """Load and encode AMASS pose data using VPoser"""
    latents = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

    if not files:
        logger.warning(f"No .npz files found in {folder_path}")
        return torch.empty(0)

    logger.info(f"Loading {len(files)} files from {folder_path}")

    for fname in tqdm(files, desc="Encoding files with VPoser"):
        try:
            file_path = os.path.join(folder_path, fname)
            data = np.load(file_path)
            pose = torch.from_numpy(data['poses'][:, 3:66]).float().to(device)
            latent = vp_model.encode(pose).mean
            latents.append(latent.cpu())
            logger.debug(f"{fname}: encoded {pose.shape[0]} frames")
        except Exception as e:
            logger.error(f"Skipped {fname}: {e}")

    final_latents = torch.cat(latents, dim=0)
    logger.info(f"All latent vectors combined. Shape: {final_latents.shape}")
    return final_latents

def create_sequences(data, seq_len):
    """Create input-target sequence pairs from data"""
    X, y = [], []
    logger.info(f"Creating sequences with window size {seq_len}...")
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    X_tensor, y_tensor = torch.stack(X), torch.stack(y)
    logger.info(f"Created {X_tensor.shape[0]} sequences. Input: {X_tensor.shape}, Target: {y_tensor.shape}")
    return X_tensor, y_tensor

def prepare_dataloader(path, vp, seq_len, batch_size, device):
    """Prepare data loader with specified sequence length and batch size"""
    latents = load_amass_pose_latents(path, vp, device)
    if latents.numel() == 0:
        logger.error("No data available to create sequences.")
        return None
    X, y = create_sequences(latents, seq_len)
    dataset = TensorDataset(X, y)
    logger.info(f"Final Dataset size: {len(dataset)} samples")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def train_and_evaluate(model, train_loader, val_loader, device, lr, epochs=15, patience=4):
    """Train model with early stopping based on validation loss"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience//2, factor=0.5)
    scaler = GradScaler() if device.type == 'cuda' else None
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses, val_losses = [], []
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        for X, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            # out = model(X)
            # loss = criterion(out, y)
            # loss.backward()
            # optimizer.step()
            if device.type == 'cuda':
                with autocast():
                    out = model(X)
                    loss = criterion(out, y)
                
                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        logger.info(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_val_loss, train_losses, val_losses

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    # Define hyperparameters to optimize
    seq_len = trial.suggest_int('seq_len', 5, 50)
    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    nhead = trial.suggest_int('nhead', 4, 16, step=4)  # Must be divisible by d_model

    if d_model % nhead != 0:
        logger.warning(f"Invalid combination: d_model={d_model}, nhead={nhead} (not divisible)")
        raise optuna.exceptions.TrialPruned()

    num_layers = trial.suggest_int('num_layers', 2, 8)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    
    logger.info(f"Trial {trial.number}: {trial.params}")
    
    # Load data with current parameters
    train_loader = prepare_dataloader(TRAIN_DIR, vp, seq_len, batch_size, DEVICE)
    val_loader = prepare_dataloader(VAL_DIR, vp, seq_len, batch_size, DEVICE)
    
    if train_loader is None or val_loader is None:
        return float('inf')
    
    # Initialize model with trial hyperparameters
    model = MinimalPoseTransformer(
        input_dim=32,
        latent_dim=32,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Train and evaluate model
    best_val_loss, _, _ = train_and_evaluate(
        model, train_loader, val_loader, DEVICE, 
        lr=lr, epochs=15, patience=4
    )
    
    # Save trial model
    os.makedirs(os.path.join(OUTPUT_DIR, 'trials'), exist_ok=True)
    torch.save({
        'trial_number': trial.number,
        'params': trial.params,
        'val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
    }, os.path.join(OUTPUT_DIR, 'trials', f'model_trial_{trial.number}.pt'))
    
    logger.info(f"Trial {trial.number} completed: val_loss={best_val_loss:.6f}")
    
    return best_val_loss

def run_hyperparameter_tuning(n_trials=50):
    """Run the complete hyperparameter tuning process"""
    # Load VPoser model
    logger.info(f"Loading VPoser model from {VPOSER_DIR}")
    global vp
    vp, _ = load_model(VPOSER_DIR, model_code=VPoser,
                     remove_words_in_model_weights='vp_model.',
                     disable_grad=True, comp_device=DEVICE)
    vp = vp.to(DEVICE)
    logger.info("VPoser model loaded successfully")
    
    # Create Optuna study
    study_name = f"pose_forecasting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage_name = f"sqlite:///{os.path.join(OUTPUT_DIR, 'study.db')}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Add suggested configurations
    logger.info("Adding baseline configuration to study")
    study.enqueue_trial({
        'seq_len': 10,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'dropout': 0.1,
        'batch_size': 256,
        'lr': 1e-4
    })
    
    # Run optimization
    logger.info(f"Starting optimization with {n_trials} trials")
    study.optimize(objective, n_trials=n_trials)
    
    # Print and save results
    logger.info("Optimization completed!")
    logger.info(f"Best trial: #{study.best_trial.number}")
    logger.info(f"Best validation loss: {study.best_trial.value:.6f}")
    logger.info("Best hyperparameters:")
    for param, value in study.best_trial.params.items():
        logger.info(f"    {param}: {value}")
    
    # Save study
    with open(os.path.join(OUTPUT_DIR, 'study.pkl'), 'wb') as f:
        pickle.dump(study, f)
    
    # Save parameter importances
    try:
        importances = optuna.importance.get_param_importances(study)
        with open(os.path.join(OUTPUT_DIR, 'param_importances.txt'), 'w') as f:
            for param, importance in importances.items():
                f.write(f"{param}: {importance:.4f}\n")
    except Exception as e:
        logger.warning(f"Could not compute parameter importances: {e}")
    
    # Train final model with best hyperparameters
    best_params = study.best_trial.params
    logger.info("Training final model with best hyperparameters")
    
    # Load data with best parameters
    train_loader = prepare_dataloader(TRAIN_DIR, vp, best_params['seq_len'], best_params['batch_size'], DEVICE)
    val_loader = prepare_dataloader(VAL_DIR, vp, best_params['seq_len'], best_params['batch_size'], DEVICE)
    test_loader = prepare_dataloader(TEST_DIR, vp, best_params['seq_len'], best_params['batch_size'], DEVICE)
    
    # Initialize model with best hyperparameters
    final_model = MinimalPoseTransformer(
        input_dim=32,
        latent_dim=32,
        d_model=best_params['d_model'],
        nhead=best_params['nhead'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    )
    
    # Train final model
    _, train_losses, val_losses = train_and_evaluate(
        final_model, train_loader, val_loader, DEVICE, 
        lr=best_params['lr'], epochs=50, patience=10
    )
    
    # Evaluate on test set
    criterion = nn.MSELoss()
    final_model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            test_loss += criterion(final_model(X), y).item()
    avg_test_loss = test_loss / len(test_loader)
    logger.info(f"Final test loss: {avg_test_loss:.6f}")
    
    # Save final model
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'hyperparameters': best_params,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': avg_test_loss
    }, os.path.join(OUTPUT_DIR, 'final_model.pt'))
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axhline(y=avg_test_loss, color='r', linestyle='--', label=f'Test Loss: {avg_test_loss:.6f}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Progress with Best Hyperparameters')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curve.png'))
    
    # Visualize results if possible
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)
        
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(OUTPUT_DIR, 'plots', 'optimization_history.png'))
        
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(OUTPUT_DIR, 'plots', 'param_importances.png'))
        
        logger.info(f"Saved visualization plots to {os.path.join(OUTPUT_DIR, 'plots')}")
    except Exception as e:
        logger.warning(f"Could not create visualization plots: {e}")
    
    logger.info("Hyperparameter tuning completed successfully!")

if __name__ == "__main__":
    run_hyperparameter_tuning(n_trials=26)
