import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from cfg import Config

class brats_dataset(Sequence):
    def __init__(self, image_list, mask_list, batch_size=Config.BATCH_SIZE):
        self.image_list, self.mask_list = image_list, mask_list
        self.batch_size = batch_size
        
    def __len__(self): 
        return len(self.image_list) // self.batch_size
        
    def __getitem__(self, idx):
        start, end = idx * self.batch_size, (idx + 1) * self.batch_size

        X = [np.load(p) for p in self.image_list[start:end]]
        y = [np.expand_dims(np.load(p).astype(np.int32), -1) for p in self.mask_list[start:end]]
        return np.array(X), np.array(y)

def get_brats_summary(loader, model):
    results = []
    for i in range(len(loader)):
        X, y_true = loader[i]
        y_pred = np.argmax(model.predict(X, verbose=0), axis=-1)
        y_true = y_true[..., 0]
        for b in range(X.shape[0]):
            scores = {}
            for name, labels in {'ET':[3], 'TC':[1,3], 'WT':[1,2,3]}.items():
                t, p = np.isin(y_true[b], labels), np.isin(y_pred[b], labels)
                dice = (2.*np.sum(t*p)+1e-6)/(np.sum(t)+np.sum(p)+1e-6)
                scores[name] = dice
            results.append(scores)
    df = pd.DataFrame(results)
    return pd.DataFrame({
        'Region': ['ET', 'TC', 'WT'],
        'Mean Dice': [df['ET'].mean(), df['TC'].mean(), df['WT'].mean()],
        'Std Dev': [df['ET'].std(), df['TC'].std(), df['WT'].std()]
    })