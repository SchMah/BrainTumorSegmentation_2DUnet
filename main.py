import os
import glob
from cfg import Config
from utils import brats_preprocessor
from model import build_unet, combined_dice_loss
from metrics import brats_dataset, get_brats_summary
from model_utils import run_training, save_final_summary
from visualization import save_training_curves, save_confusion_matrix, generate_visual_samples
import tensorflow as tf

def main():
    
    all_dirs = sorted([os.path.join(Config.DATASET_ROOT, d) for d in os.listdir(Config.DATASET_ROOT) 
                       if os.path.isdir(os.path.join(Config.DATASET_ROOT, d))])
    
    if Config.preprocessing:
        print("preprocessing...")
        pre = brats_preprocessor(Config.TRAIN_SAVE_DIR)
        pre.process_and_save(all_dirs[:Config.TRAIN_PATIENTS])
        
        test_pool = all_dirs[Config.TEST_PATIENTS_RANGE[0]:Config.TEST_PATIENTS_RANGE[1]]
        pre_test = brats_preprocessor(Config.TEST_SAVE_DIR)
        pre_test.process_and_save(test_pool)

    # split train to train/val
   
    imgs = sorted(glob.glob(f"{Config.TRAIN_SAVE_DIR}/images/*.npy"))
    msks = sorted(glob.glob(f"{Config.TRAIN_SAVE_DIR}/masks/*.npy"))
    split = int(len(imgs) * 0.8) 
    
    train_set = brats_dataset(imgs[:split], msks[:split])
    val_set = brats_dataset(imgs[split:], msks[split:])

    # test data 
    test_imgs = sorted(glob.glob(f"{Config.TEST_SAVE_DIR}/images/*.npy"))
    test_msks = sorted(glob.glob(f"{Config.TEST_SAVE_DIR}/masks/*.npy"))
    test_set  = brats_dataset(test_imgs, test_msks) 
    
    model = build_unet()


    if os.path.exists(Config.MODEL_PATH):
        print(f"loading weights from {Config.MODEL_PATH}...")
        model.load_weights(Config.MODEL_PATH)
    elif not Config.DO_TRAINING:
        print("no weight found and DO_TRAINING is False..")
        return


    
    if Config.train:
        model.compile(optimizer='adam',
                      loss= combined_dice_loss,
                      metrics=['accuracy'])
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(Config.MODEL_PATH, save_best_only=True),
            tf.keras.callbacks.CSVLogger(Config.log_file, append=True),
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
        ]
        history = run_training(model, train_set, val_set, callbacks)

    if Config.report:
        print("\n Visualization and Metrics")
        save_training_curves(Config.log_file)
        save_confusion_matrix(test_set, model)
        generate_visual_samples(test_set, model)
        summary = get_brats_summary(test_set, model)
        save_final_summary(summary)
        print(summary)


if __name__ == '__main__':
    main()
