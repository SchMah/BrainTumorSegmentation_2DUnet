import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from cfg import Config
from metrics import get_brats_summary
import matplotlib.patches as mpatches

def save_training_curves(log_path, save_path='results/curves.png'):
    """Plots and saves the loss/accuracy curves from the CSVLogger."""
    df = pd.read_csv(log_path)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['loss'], label='Train')
    plt.plot(df['val_loss'], label='Val')
    plt.title('Loss History')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(df['accuracy'], label='Train')
    plt.plot(df['val_accuracy'], label='Val')
    plt.title('Accuracy History')
    plt.legend()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_confusion_matrix(loader, model, save_path='results/cm.png'):
    """Plots confusion matrix."""
    y_true_all = []
    y_pred_all = []
    
    # We'll take a few batches for the CM to avoid memory issues
    for i in range(min(5, len(loader))):
        X, y = loader[i]
        preds = model.predict(X, verbose=0)
        y_true_all.extend(y.flatten())
        y_pred_all.extend(np.argmax(preds, axis=-1).flatten())
        
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2, 3])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=Config.CLASS_NAMES, yticklabels=Config.CLASS_NAMES)
    plt.title('Normalized Confusion Matrix')
    plt.savefig(save_path)
    plt.close()



def generate_visual_samples(loader, model, save_path='results/samples.png'):
    """Generates an example of model evaluation."""
    
   
    for batch_idx in range(len(loader)):
        X, y = loader[batch_idx]
        
       
        if np.sum(y[..., 0] > 0) > 500: 
            
            
            preds = model.predict(X, verbose=0)
            preds_max = np.argmax(preds, axis=-1)
            
            
            tumor_sums = np.sum(y[..., 0] > 0, axis=(1, 2)) 
            top_3_indices = np.argsort(tumor_sums)[-3:][::-1] 
            
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for i, slice_idx in enumerate(top_3_indices):
                
                axes[0, i].imshow(X[slice_idx, ..., 3], cmap='gray') 
                axes[0, i].imshow(np.ma.masked_where(y[slice_idx, ..., 0] == 0, y[slice_idx, ..., 0]), cmap='jet', alpha=0.5)
                axes[0, i].set_title(f"Ground Truth")
                
                
                axes[1, i].imshow(X[slice_idx, ..., 3], cmap='gray')
                axes[1, i].imshow(np.ma.masked_where(preds_max[slice_idx] == 0, preds_max[slice_idx]), cmap='jet', alpha=0.5)
                axes[1, i].set_title(f"Prediction")
                
            for ax in axes.flatten(): ax.axis('off')
            plt.savefig(save_path)
            plt.close()
            
            print(f"Visual samples generated from batch {batch_idx}.")
            return 
            

