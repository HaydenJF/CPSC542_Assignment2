import tensorflow as tf

class ImageAugmenter:
    def __init__(self, image_size):
        self.image_size = image_size

    def augment(self, image, mask):
        # Applying random horizontal flipping
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

        image = tf.image.random_brightness(image, max_delta=0.2)

        return image, mask
