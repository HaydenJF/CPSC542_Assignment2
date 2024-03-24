import tensorflow as tf
import tensorflow_datasets as tfds
from augmentation import ImageAugmenter

import matplotlib.pyplot as plt

def create_data():
    dataset_train, dataset_val, dataset_test = load_dataset()
    dataset_train, dataset_val, dataset_test = preprocess_datasets(dataset_train, dataset_val, dataset_test)
    return dataset_train, dataset_val, dataset_test

def load_dataset():
    (dataset_train, dataset_val, dataset_test), dataset_info = tfds.load(
        'oxford_iiit_pet:3.*.*',
        split=['train[:80%]', 'train[80%:]', 'test'],  # Use 80% of train for training and the rest for validation
        with_info=True,
        as_supervised=False  # This loads the dataset in a dictionary format
    )
    return dataset_train, dataset_val, dataset_test

def preprocess_datasets(dataset_train, dataset_val, dataset_test):
    IMG_SIZE = 128  # Set the target image size
    augmenter = ImageAugmenter(IMG_SIZE)  # Instantiate the augmenter

    def preprocess_and_augment(data):
        image = data['image']
        mask = data['segmentation_mask']
        
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE],
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )

        image = tf.cast(image, tf.float32) / 255.0
        mask -= 1
        
        image, mask = augmenter.augment(image, mask)  # Apply augmentation
        return image, mask

    BATCH_SIZE = 8

    dataset_train = dataset_train.map(preprocess_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_train = dataset_train.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    dataset_val = dataset_val.map(preprocess_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_val = dataset_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    dataset_test = dataset_test.map(preprocess_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_test = dataset_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset_train, dataset_val, dataset_test


def display_sample_images(dataset, samples=5):
    plt.figure(figsize=(10, 5 * samples))  # Increase width if titles are getting cut off
    titles = ['Input Image', 'True Mask']

    # Enumerate the samples, get the first element from each batch (image, mask)
    for i, (image, mask) in enumerate(dataset.unbatch().take(samples)):
        image = image.numpy()
        mask = mask.numpy().squeeze()  # Assuming mask is (height, width, 1)

        # Display image
        plt.subplot(samples, 2, i * 2 + 1)
        plt.title(titles[0])
        plt.axis('off')
        plt.imshow(image)

        # Display mask
        plt.subplot(samples, 2, i * 2 + 2)
        plt.title(titles[1])
        plt.axis('off')
        plt.imshow(mask, cmap='BuPu')  # Use cmap='gray' to show the mask in grayscale

    plt.tight_layout()  # This will adjust spacing between plots to prevent title overlap
    plt.show()



    
def check_image_dimensions(dataset, expected_shape_images=(128, 128, 3), expected_shape_masks=(128, 128, 1)):
    for images, masks in dataset.take(1):  # Take 1 batch of images and masks
        for i in range(images.shape[0]):  # Iterate through the batch
            image_shape = images[i].shape
            if image_shape != expected_shape_images:
                print(f"Image at index {i} has shape {image_shape}, expected {expected_shape_images}.")
        for i in range(masks.shape[0]):  # Iterate through the batch
            mask_shape = masks[i].shape
            if mask_shape != expected_shape_masks:
                print(f"Image at index {i} has shape {mask_shape}, expected {expected_shape_masks}.")