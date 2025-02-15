import os
import numpy as np
import nibabel as nib
import cv2
import random
from cfg import Config

def load_patient_volume(patient_path):
    patient_id = os.path.basename(patient_path)
    paths = {
        't1n': f"{patient_id}-t1n.nii.gz",
        't1c': f"{patient_id}-t1c.nii.gz",
        't2w': f"{patient_id}-t2w.nii.gz",
        't2f': f"{patient_id}-t2f.nii.gz",
        'seg': f"{patient_id}-seg.nii.gz"
    }
    
    img_data = []
    for m in ['t1n', 't1c', 't2w', 't2f']:
        img_data.append(nib.load(os.path.join(patient_path, paths[m])).get_fdata())
    
    mask = nib.load(os.path.join(patient_path, paths['seg'])).get_fdata()
    image_data = np.stack(img_data, axis=-1)
    return image_data, mask

class brats_preprocessor:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "masks"), exist_ok=True)

    def normalize(self, volume):
        mask_brain = volume > 0 
        for c in range(4):
            channel_data = volume[..., c]
            if np.sum(mask_brain[..., c]) > 0:
                mean = np.mean(channel_data[mask_brain[..., c]])
                std = np.std(channel_data[mask_brain[..., c]])
                volume[..., c] = (channel_data - mean) / (std + 1e-8)
        return volume

    def crop_and_resize(self, img, is_mask=False):
        c = Config.CROP_LIMITS
        crop_img = img[c[0]:c[1], c[0]:c[1]]
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.resize(crop_img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=interp)

    def process_and_save(self, patient_dirs):
        slice_counter = 0
        for path in patient_dirs:
            try:
                img_vol, mask_vol = load_patient_volume(path)
                img_vol = self.normalize(img_vol)
                for i in range(Config.SLICE_RANGE[0], Config.SLICE_RANGE[1]):
                    slice_mask = mask_vol[:, :, i]
                    slice_img = img_vol[:, :, i, :]
                    if np.sum(slice_mask) > 0 or (random.random() < 0.05 and np.sum(slice_img) > 0.1):
                        p_img = np.stack([self.crop_and_resize(slice_img[..., c]) for c in range(4)], axis=-1)
                        p_mask = self.crop_and_resize(slice_mask, is_mask=True)
                        np.save(f"{self.save_dir}/images/img_{slice_counter}.npy", p_img)
                        np.save(f"{self.save_dir}/masks/mask_{slice_counter}.npy", p_mask)
                        slice_counter += 1
            except Exception as e:
                print(f"error {path}: {e}")