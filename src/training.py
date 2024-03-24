import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import kerastuner as kt

from datetime import datetime
import matplotlib.pyplot as plt
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'


def build_unet(hp):
    input_shape = (128, 128, 3)  # Adjust based on your dataset
    inputs = Input(shape=input_shape)

    # Hyperparameters
    num_blocks = hp.Int('num_blocks', min_value=2, max_value=4, step=1)
    initial_num_filters = hp.Int('initial_num_filters', min_value=32, max_value=128, step=32)

    x = inputs
    skips = []
    for i in range(num_blocks):
        # Encoder
        num_filters = initial_num_filters * (2 ** i)
        x = conv_block(x, num_filters)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)

    # Bottleneck
    x = conv_block(x, num_filters * 2)

    # Decoder
    for i in reversed(range(num_blocks)):
        num_filters = initial_num_filters * (2 ** i)
        x = UpSampling2D((2, 2))(x)
        skip = skips.pop()
        x = Concatenate()([x, skip])
        x = conv_block(x, num_filters)

    # Output layer
    outputs = Conv2D(3, 1, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

    return model

def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    return x

def run_tuner(dataset_train, dataset_val, c):
    tuner = kt.RandomSearch(
        build_unet,
        objective='val_accuracy',
        max_trials=c["max_trials"],  # Adjust as necessary
        executions_per_trial=c["executions_per_trial"],  # Adjust as necessary for reliability
        directory='../models',
        project_name='unet_tuning'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=c["patience"])

    tuner.search(
        dataset_train,
        epochs=c["epochs"],  # Adjust epochs according to your need
        validation_data=dataset_val,
        callbacks=[stop_early]
    )

    return tuner

def get_best_model(dataset_train, dataset_val):
    '''c = {
        "max_trials": 4,
        "executions_per_trial": 1,
        "epochs": 10,
        "patience": 3
        "m": 5
    }'''
    c = {
        "max_trials": 4,
        "executions_per_trial": 1,
        "epochs": 2,
        "patience": 1,
        "m": 1
    }
    
    tuner = run_tuner(dataset_train, dataset_val, c)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = build_unet(best_hps)
    
    stop_early = EarlyStopping(monitor='val_loss', patience=c["patience"]*c["m"])

    history = best_model.fit(
        dataset_train,
        epochs=c["epochs"]*c["m"],  # Train for more epochs
        validation_data=dataset_val,
        callbacks=[stop_early]
    )

    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    best_model.save(f'../models/unet_model_{datetime_str}.weights.h5')

    return best_model, history

def evaluate_model_performance(model, dataset_val):
    val_loss, val_accuracy = model.evaluate(dataset_val)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")

import matplotlib.pyplot as plt

def plot_training_history(history, metric='accuracy', val_metric='val_accuracy', 
                          loss='loss', val_loss='val_loss', title_suffix=''):
    # Plot specified metric values
    plt.figure(figsize=(12, 5))
    
    # Plot for the provided metric
    plt.subplot(1, 2, 1)
    plt.plot(history.history[metric])
    plt.plot(history.history[val_metric])
    plt.title(f'Model {metric.capitalize()} {title_suffix}')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot for the loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history[loss])
    plt.plot(history.history[val_loss])
    plt.title(f'Model Loss {title_suffix}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()
