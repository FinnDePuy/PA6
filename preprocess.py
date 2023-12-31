from keras.utils import image_dataset_from_directory
from config import train_directory, test_directory, image_size, batch_size, validation_split
import tensorflow as tf

def _split_data(train_directory, test_directory, batch_size, validation_split):
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )
    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset

def _augment_dataset(dataset):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        # tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1)
    ])
    augmented_dataset = dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return augmented_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split)
    # YOUR CODE
    # call augment_dataset
    train_dataset = _augment_dataset(train_dataset)


    return train_dataset, validation_dataset, test_dataset