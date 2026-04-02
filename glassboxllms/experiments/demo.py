import logging
from typing import Any
import os
from datetime import datetime
import time
import math

from glassboxllms.runner.config import Config
from glassboxllms.runner.tracking import Tracker


def run_experiment(cfg: Config, model: Any, dataset: Any, tracker: Tracker):
    logging.info("DUMMY || DUMMY || DUMMY || DUMMY")
    logging.info(
        "This is a demo experiment to demonstrate the runner/config functionality with realistic ML training simulation."
    )

    # here's accessing information about the model
    logging.info(f"Model: {cfg.model.name} (Checkpoint: {cfg.model.checkpoint})")
    logging.info(f"Device: {cfg.model.device}, Dtype: {cfg.model.dtype}")

    # here's accessing information about the dataset
    logging.info(f"Dataset: {cfg.dataset.path} (Split: {cfg.dataset.split})")
    if cfg.dataset.preprocess:
        logging.info(f"Preprocessing: {cfg.dataset.preprocess}")

    # accessing tracker information
    logging.info(f"Tracking Enabled: {cfg.tracking.enabled}")
    if cfg.tracking.enabled:
        logging.info(f"Tracker Type: {cfg.tracking.type}, Project: {cfg.tracking.project}")

    # here's accessing arbitrary parameters provided
    # in the config's experiment's parameters key
    # note including experiment-specific parameters in
    # the 'parameters' key is recommended good practice!
    lr = cfg.experiment.parameters.get("lr", 2e-5)
    bsz = cfg.experiment.parameters.get("bsz", 8)
    num_epochs = cfg.experiment.parameters.get("epochs", 10)
    logging.info(f"Experiment parameters: lr={lr}, bsz={bsz}, epochs={num_epochs}")

    # Simulate a realistic training run with loss curves
    logging.info("Starting simulated training run...")

    # Simulate training data
    num_samples = 1000
    steps_per_epoch = num_samples // bsz

    # Initialize metrics for tracking
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        logging.info(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        current_lr = lr * (0.95 ** epoch)

        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_val_loss = 0.0
        epoch_val_correct = 0

        for step in range(steps_per_epoch):
            base_loss = 2.5 * math.exp(-0.3 * (epoch + step / steps_per_epoch))
            noise = (math.sin(step * 0.5) + math.cos(epoch * 0.3)) * 0.1
            train_loss = max(0.1, base_loss + noise)

            base_acc = 0.85 * (1 - math.exp(-0.5 * (epoch + step / steps_per_epoch)))
            acc_noise = (math.sin(step * 0.3) - math.cos(epoch * 0.2)) * 0.02
            train_acc = max(0.5, min(0.98, base_acc + acc_noise))

            epoch_train_loss += train_loss
            epoch_train_correct += int(train_acc * bsz)

            if (step + 1) % 10 == 0 or step == 0:
                global_step = epoch * steps_per_epoch + step
                metrics = {
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "train/learning_rate": current_lr,
                    "epoch": epoch + 1,
                }
                tracker.log(metrics, step=global_step)

            # Simulate computation time
            time.sleep(0.01)

        # Calculate epoch averages
        avg_train_loss = epoch_train_loss / steps_per_epoch
        avg_train_acc = epoch_train_correct / (steps_per_epoch * bsz)

        # Simulate validation (slightly different metrics)
        avg_val_loss = avg_train_loss * 1.05 + 0.02
        avg_val_acc = avg_train_acc * 0.98

        # Store for final summary
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)

        # Log epoch summary at the end of the epoch's global step
        final_global_step = (epoch + 1) * steps_per_epoch - 1
        epoch_metrics = {
            "train/epoch_loss": avg_train_loss,
            "train/epoch_accuracy": avg_train_acc,
            "val/epoch_loss": avg_val_loss,
            "val/epoch_accuracy": avg_val_acc,
            "learning_rate": current_lr,
        }
        tracker.log(epoch_metrics, step=final_global_step)

        logging.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        logging.info(f"           Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

    # Final summary metrics
    final_metrics = {
        "final/train_loss": train_losses[-1],
        "final/val_loss": val_losses[-1],
        "final/train_accuracy": train_accs[-1],
        "final/val_accuracy": val_accs[-1],
        "best/val_accuracy": max(val_accs),
        "best/val_loss": min(val_losses),
        "total_epochs": num_epochs,
    }
    tracker.log(final_metrics)

    logging.info("\n=== Training Complete ===")
    logging.info(f"Final Training Loss: {train_losses[-1]:.4f}")
    logging.info(f"Final Validation Loss: {val_losses[-1]:.4f}")
    logging.info(f"Final Training Accuracy: {train_accs[-1]:.4f}")
    logging.info(f"Final Validation Accuracy: {val_accs[-1]:.4f}")
    logging.info(f"Best Validation Accuracy: {max(val_accs):.4f}")

    # here's how you can make it output things
    output_path = os.path.join(cfg.output.base_dir, cfg.output.name)
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "results.txt"), "w") as f:
        f.write("== DEMO EXPERIMENT RUN ==\n")
        f.write(str(datetime.now()) + "\n\n")
        f.write(f"Experiment Configuration:\n")
        f.write(f"  Learning Rate: {lr}\n")
        f.write(f"  Batch Size: {bsz}\n")
        f.write(f"  Epochs: {num_epochs}\n")
        f.write(f"  Samples per Epoch: {num_samples}\n\n")
        f.write(f"Training Results:\n")
        f.write(f"  Final Train Loss: {train_losses[-1]:.4f}\n")
        f.write(f"  Final Val Loss: {val_losses[-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {train_accs[-1]:.4f}\n")
        f.write(f"  Final Val Accuracy: {val_accs[-1]:.4f}\n")
        f.write(f"  Best Val Accuracy: {max(val_accs):.4f}\n")
        f.write(f"  Best Val Loss: {min(val_losses):.4f}\n\n")
        f.write(f"Epoch-by-Epoch Summary:\n")
        for i, (tl, vl, ta, va) in enumerate(zip(train_losses, val_losses, train_accs, val_accs)):
            f.write(f"  Epoch {i+1}: Train Loss={tl:.4f}, Val Loss={vl:.4f}, Train Acc={ta:.4f}, Val Acc={va:.4f}\n")

    logging.info(f"Experiment summary saved to {output_path}")
    logging.info("--- Dummy experiment completed! ---")
