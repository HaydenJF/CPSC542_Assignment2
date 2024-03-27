from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
#from GradCAM import GradCAM
#import cv2

def evaluate_model(model, dataset):
    loss, accuracy = model.evaluate(dataset)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')
    
    TP_list = []
    TN_list = []
    FP_list = []
    FN_list = []

    for batch in dataset.take(30):
        x_batch, y_true_batch = batch
        y_pred_batch = model.predict(x_batch)
        
        for i in range(len(x_batch)):
            y_true = y_true_batch[i].numpy()
            y_pred = (y_pred_batch[i] > 0.5).astype(int)

            TP_list.append(np.sum((y_true == 1) & (y_pred == 1)))
            TN_list.append(np.sum((y_true == 0) & (y_pred == 0)))
            FP_list.append(np.sum((y_true == 0) & (y_pred == 1)))
            FN_list.append(np.sum((y_true == 1) & (y_pred == 0)))

    TP_avg = np.mean(TP_list)
    TN_avg = np.mean(TN_list)
    FP_avg = np.mean(FP_list)
    FN_avg = np.mean(FN_list)

    precision_avg = TP_avg / (TP_avg + FP_avg) if (TP_avg + FP_avg) > 0 else 0
    recall_avg = TP_avg / (TP_avg + FN_avg) if (TP_avg + FN_avg) > 0 else 0
    f1_score_avg = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg) if (precision_avg + recall_avg) > 0 else 0
    
    array = [TN_avg, FP_avg, FN_avg, TP_avg, loss, accuracy, precision_avg, recall_avg, f1_score_avg]
    return array

def display_scores(array):
    TN_avg = array[0]
    FP_avg = array[1]
    FN_avg = array[2]
    TP_avg = array[3]
    loss = array[4]
    accuracy = array[5]
    precision_avg = array[6]
    recall_avg = array[7]
    f1_score_avg = array[8]
    
    
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')
    print("Average Precision:", precision_avg)
    print("Average Recall:", recall_avg)
    print("Average F1 Score:", f1_score_avg)

    confusion_matrix_avg = np.array([[TP_avg, FN_avg], [FP_avg, TN_avg]])

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix_avg, annot=True, fmt='.2f', cmap='Blues', ax=ax)

    label_names = ['Positive', 'Negative']
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Average Confusion Matrix')
    ax.xaxis.set_ticklabels(label_names)
    ax.yaxis.set_ticklabels(label_names, rotation=0)
    plt.show()

    total_avg = TN_avg + FP_avg + FN_avg + TP_avg
    confusion_matrix_fraction = np.array([[TP_avg/total_avg, FN_avg/total_avg], [FP_avg/total_avg, TN_avg/total_avg]])

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix_fraction, annot=True, fmt='.2f', cmap='Blues', ax=ax)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Decimal Average Confusion Matrix')
    ax.xaxis.set_ticklabels(label_names)
    ax.yaxis.set_ticklabels(label_names, rotation=0)
    plt.show()

def predict(model, dataset):
    predictions = model.predict(dataset)
    predictions = predictions > 0.5
    return predictions

def three_best_three_worst_predictions(model, dataset):
    differences = []
    for batch in dataset.take(30):  # Adjust the number as needed
        x_batch, y_true_batch = batch
        y_pred_batch = model.predict(x_batch)
        
        for i in range(len(x_batch)):
            x = x_batch[i]
            y_true = y_true_batch[i]
            y_pred = y_pred_batch[i]
            print("y_true", type(y_true))
            print("y_pred", type(y_pred))
            
            y_true_np = y_true.numpy()
            y_pred = y_pred > 0.5

            TP = np.sum((y_true_np == 1) & (y_pred == 1))
            TN = np.sum((y_true_np == 0) & (y_pred == 0))
            FP = np.sum((y_true_np == 0) & (y_pred == 1))
            FN = np.sum((y_true_np == 1) & (y_pred == 0))
            print("TP: ", TP)
            print("TN: ", TN)
            print("FP: ", FP)
            print("FN: ", FN)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1_score)
            difference = np.abs(y_pred - y_true.numpy())
            differences.append((x, y_true, y_pred, np.sum(difference)))
            
            return None

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
    return evaluate_model(model, test_dataset)
    '''
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
    '''
        