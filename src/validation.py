from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from GradCAM import GradCAM
import cv2

def evaluate_model(model, dataset):
    loss, accuracy = model.evaluate(dataset)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')


def three_best_three_worst_predictions(model, dataset):
    differences = []
    for batch in dataset.take(30):  # Adjust the number as needed
        x_batch, y_true_batch = batch
        y_pred_batch = model.predict(x_batch)
        
        for i in range(len(x_batch)):
            x = x_batch[i]
            y_true = y_true_batch[i]
            y_pred = y_pred_batch[i]
            difference = np.abs(y_pred - y_true.numpy())
            differences.append((x, y_true, y_pred, np.sum(difference)))

    differences.sort(key=lambda x: x[3])
    best_predictions = differences[:3]  # Take the 3 best predictions
    worst_predictions = differences[-3:]  # Take the 3 worst predictions
    
    return best_predictions, worst_predictions

def find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        # Check if the layer is a convolutional layer
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None  # Return None if no convolutional layer is found

def apply_gradcam(img, model, gradcam, layerName=None):
    img_array = np.expand_dims(img, axis=0)  # Convert to 4D array; this is necessary for model prediction
    heatmap = gradcam.compute_heatmap(img_array, eps=1e-8)  # Compute heatmap; removed the 'model' and 'layerName' parameters as they are already part of the GradCAM class
    # Resize heatmap to the original image size and overlay
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    (heatmap, output) = gradcam.overlay_heatmap(heatmap, img, alpha=0.5)
    plt.imshow(output)  # Display the overlaid image
    plt.axis('off')
    plt.show()

def validation(model, test_dataset, layerName=None):
    evaluate_model(model, test_dataset)
    
    # Initialize GradCAM
    if layerName is None:
        layerName = GradCAM(model, 0, 'conv2d_18').find_target_layer()
    gradcam = GradCAM(model, 0, layerName)

    # Get best and worst predictions
    best_predictions, worst_predictions = three_best_three_worst_predictions(model, test_dataset)

    # Apply GradCAM to best and worst predictions
    for img, _, _ in best_predictions + worst_predictions:
        heatmap = gradcam.compute_heatmap(np.expand_dims(img, axis=0))
        heatmap, output = gradcam.overlay_heatmap(heatmap, img)
        plt.imshow(output)
        plt.show()
        
        