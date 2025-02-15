import os
import csv
import tensorflow as tf
from cfg import Config

def run_training(model, train_loader, val_loader, callbacks):

    print("\n" + "="*50)
    print("2D TRAINING SUMMARY")
    print(f"Data Directory:  {Config.TRAIN_SAVE_DIR}")
    print(f"Model Name:      {model.name}")
    print(f"Learning Rate:   {Config.LEARNING_RATE}")
    print(f"Batch Size:      {Config.BATCH_SIZE}")
    print(f"Max Epochs:      {Config.EPOCHS}")
    print(f"Target Size:     {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print("="*50 + "\n")

    try:
        history = model.fit(
            train_loader,
            validation_data=val_loader,
            epochs=Config.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        print("\ntraining completed successfully.")
        return history
    except Exception as e:
        print(f"\ntraining failed: {e}")
        return None

def save_final_summary(summary_df, path="results/final_metrics.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    summary_df.to_csv(path, index=False)
    print(f"Final metrics saved to {path}")