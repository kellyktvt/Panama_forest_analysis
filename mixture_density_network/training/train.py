import os
import argparse
import torch
import optuna
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from mixture_density_network.src.blocks import MixtureDensityNetwork, NoiseType
from mixture_density_network.data_processing.data import MyDataModule


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Get interests from filename
        filename = os.path.basename(self.args.data_file).replace(".pt", "")
        interest_str = filename.replace("dataset_", "").split("_rarity")[0]
        interests = interest_str.split("_")
        output_dim = len(interests)

        # Define the model
        self.model = MixtureDensityNetwork(1024, output_dim, n_components=self.args.num_components, hidden_dim=self.args.hidden_dim, noise_type=NoiseType.DIAGONAL)

    def forward(self, X_batch):
        if isinstance(X_batch, (list, tuple)):
            X_batch = X_batch[0] if len(X_batch) == 1 else torch.cat(X_batch, dim=0)
        return self.model(X_batch)
    
    def compute_loss(self, X_batch, y_batch):
        """ Compute the MDN negative log-likelihood loss. """
        return self.model.loss(X_batch, y_batch).mean()

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        loss = self.compute_loss(X_batch, y_batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        loss = self.compute_loss(X_batch, y_batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        optimizer = AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay, betas=(self.args.beta1, self.args.beta2))
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=steps_per_epoch)
        cosine = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs * steps_per_epoch, eta_min=self.args.learning_rate*0.1)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5 * steps_per_epoch])  # milestone is when to switch from warm up to decay (after 5 epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    

# Run training
def train(args):
    print("Running: Training")
    print("Loading model and dataset...")
    model = LitModel(args)
    datamodule = MyDataModule(args.data_file, args.batch_size)
    
    # Get interests from filename
    filename = os.path.basename(args.data_file).replace(".pt", "")
    interest_str = filename.replace("dataset_", "").split("_rarity")[0]
    interests = interest_str.split("_")
    interest_str = "_".join(interests)

    # Get rarity threshold if present
    rarity_threshold = None
    if "_rarity" in filename:
        rarity_threshold = filename.split("_rarity")[-1]
    
    # Create output directory
    if rarity_threshold:
        save_dir = os.path.join(args.output, interest_str, f"batch{args.batch_size}_epochs{args.epochs}_lr{args.learning_rate}_weightdecay{args.weight_decay}_beta1{args.beta1}_beta2{args.beta2}_dropout{args.dropout}_numcomponents{args.num_components}_hidden{args.hidden_dim}_rarity{rarity_threshold}")
    else:
        save_dir = os.path.join(args.output, interest_str, f"batch{args.batch_size}_epochs{args.epochs}_lr{args.learning_rate}_weightdecay{args.weight_decay}_beta1{args.beta1}_beta2{args.beta2}_dropout{args.dropout}_numcomponents{args.num_components}_hidden{args.hidden_dim}")
    os.makedirs(save_dir, exist_ok=True)

    # Create checkpoints directory
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Lightning callbacks and logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=f"checkpoint_{{epoch:02d}}_{{val_loss:.2f}}",
        monitor="val_loss",
        save_top_k=3,
        mode="min"
    )
    logger = CSVLogger(save_dir=save_dir, name="training", version=None)
        
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.cuda_num],
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        # accumulate_grad_batches=4,          # simulate a 4× larger batch size (so 3x4=16)
        max_epochs=args.epochs,
        # enable_checkpointing=False,
        gradient_clip_val=1.0,              # Clip gradients if they exceed 1.0
        logger=logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", patience=20, mode="min")],
    )
    # Train the model
    trainer.fit(model, datamodule)

    # Save Lightning checkpoint as plain PyTorch state_dict for plotting
    if checkpoint_callback.best_model_path:
        ckpt = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
        torch.save(ckpt['state_dict'], os.path.join(save_dir, "mdn_model_original.pth"))
        print(f"Saved model state_dict to {os.path.join(save_dir, 'mdn_model_original.pth')}")


# Run tuning
def objective(trial, args, save_dir):
    # Sample hyperparameters
    num_components = trial.suggest_categorical("num_components", [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # Clone args and override
    trial_args = argparse.Namespace(**vars(args))
    trial_args.num_components = num_components

    model = LitModel(trial_args)
    datamodule = MyDataModule(trial_args.data_file, args.batch_size)

    # Create checkpoints directory
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=f"checkpoint_{{epoch:02d}}_{{val_loss:.2f}}",
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[trial_args.cuda_num],
        max_epochs=trial_args.epochs,
        enable_progress_bar=False,
        logger=False,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", patience=5, mode="min")],
    )
    trainer.fit(model, datamodule)
    val_result = trainer.validate(model, datamodule)
    return val_result[0]["val_loss"]
    
  
def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments.")
    parser.add_argument('-d', '--data_file', type=str, required=True, help='Path to dataset_*.pt file')
    parser.add_argument("-o", "--output", type=str, required=True, help="Output path.")
    # parser.add_argument("--checkpoint_path", type=str, default="esm2_t6_8M_UR50D", help="Model checkpoint name or full path.")
    parser.add_argument("--num_components", type=int, default=5, help="Number of components in MDN.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--cuda_num", type=int, help="Which GPU to use (0,1,...) or -1 for all available")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1.")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2.")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search instead of standard training.")
    parser.add_argument("--dropout", type=float, default=0.1, help="CLS dropout probability.")
    return parser.parse_args()
  

if __name__ == '__main__':
    args = parse_args()
    
    if args.tune:
        # Get interests from filename
        filename = os.path.basename(args.data_file).replace(".pt", "")
        interest_str = filename.replace("dataset_", "").split("_rarity")[0]
        interests = interest_str.split("_")
        interest_str = "_".join(interests)
        
        # Create output directory
        save_dir = os.path.join(args.output, interest_str, "tuning")
        os.makedirs(save_dir, exist_ok=True)

        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, args, save_dir), n_trials=50)

        print("\n Best trial:")
        print(f"  Value: {study.best_trial.value}")
        print(f"  Params: {study.best_trial.params}")

    else:
        # Load best Optuna parameters if they exist
        filename = os.path.basename(args.data_file).replace(".pt", "")
        interest_str = filename.replace("dataset_", "").split("_rarity")[0]
        interests = interest_str.split("_")
        interest_str = "_".join(interests)

        best_params_path = os.path.join(args.output, interest_str, "tuning", "best_params.pt")

        if os.path.exists(best_params_path):
            print(f"Found best params at {best_params_path}")
            best_params = torch.load(best_params_path)
            print("Using tuned params:", best_params)

            # Override only the tuned hyperparameters
            for key, val in best_params.items():
                if hasattr(args, key):
                    setattr(args, key, val)
            print("Updated args with tuned params.")
        else:
            print("No best_params.pt found — using default args.")

        # Proceed to training
        train(args)

