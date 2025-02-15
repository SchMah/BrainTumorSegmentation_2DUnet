import os

class Config:

    DATASET_ROOT = 'TrainingData'
    TRAIN_SAVE_DIR = "2D_Dataset"
    TEST_SAVE_DIR = "Test_Dataset"
    MODEL_PATH = 'best_model.keras'
    log_file = 'training_log.csv'
    
    IMG_SIZE = 128
    CROP_LIMITS = (20, 220)
    SLICE_RANGE = (20, 135)
    TRAIN_PATIENTS = 500
    TEST_PATIENTS_RANGE = (501, 610)

    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 4
    CLASS_NAMES = ['Background', 'Necrotic Core', 'Edema', 'Enhancing Tumor']

    preprocessing = False  
    train      = False 
    report     = True 