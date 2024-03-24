import tensorflow as tf
import numpy as np


def predict(model, dataset):
    predictions = model.predict(dataset)
    predictions = predictions > 0.5
    return predictions

def display_predictions(display_list, titles=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(titles[i] if titles else f'Image {i}')
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def show_best_worst_predictions(model, dataset):
    differences = []
    # Iterate through batches in the dataset
    for batch in dataset.take(30):  # Adjust the number as needed
        x_batch, y_true_batch = batch
        y_pred_batch = model.predict(x_batch)
        
        # Iterate through each item in the batch
        for i in range(len(x_batch)):
            x = x_batch[i]
            y_true = y_true_batch[i]
            y_pred = y_pred_batch[i]
            
            # Calculate the difference for this item
            difference = np.abs(y_pred - y_true.numpy())  # Adjust this line as needed for your use case
            differences.append((x, y_true, y_pred, np.sum(difference)))

    # Sort by difference to find the best and worst predictions
    differences.sort(key=lambda x: x[3])

    # Display the 3 best predictions
    print("Best Predictions:")
    for x, y_true, y_pred, _ in differences[:3]:
        display_predictions([x, y_true, y_pred > 0.5], titles=['Input', 'True', 'Predicted'])

    # Display the 3 worst predictions
    print("Worst Predictions:")
    for x, y_true, y_pred, _ in differences[-3:]:
        display_predictions([x, y_true, y_pred > 0.5], titles=['Input', 'True', 'Predicted'])


def prediction(model, test_dataset):
    test_predictions = predict(model, test_dataset)
    display_predictions(test_predictions)
    show_best_worst_predictions(model, test_dataset)