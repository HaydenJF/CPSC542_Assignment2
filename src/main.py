import warnings
warnings.filterwarnings('ignore')
import importlib
import tensorflow as tf

# Ensure this runs before any TensorFlow operations
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("here")
        print(e)

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'


import preprocessing as preprocessing
importlib.reload(preprocessing)

dataset_train, dataset_val, dataset_test = preprocessing.create_data()

preprocessing.display_sample_images(dataset_train, samples=5)
#preprocessing.display_sample_images(dataset_val, samples=5)

#preprocessing.check_image_dimensions(dataset_train)
#preprocessing.check_image_dimensions(dataset_val)

import training
importlib.reload(training)

best_model, history = training.get_best_model(dataset_train, dataset_val)

training.evaluate_model_performance(best_model, dataset_val)
training.plot_training_history(history)

