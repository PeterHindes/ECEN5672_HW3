import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# print("TensorFlow Version:", tf.__version__)
# print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# Define the CNN model
model = Sequential(
    [
        # Encoder
        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            input_shape=(28, 28, 1),
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.Flatten(),
        layers.Dense(
            24, kernel_initializer="he_normal", name="encoder_bottleneck_output"
        ),
        layers.LeakyReLU(alpha=0.3),
        # Decoder
        layers.Dense(3136, kernel_initializer="he_normal"),
        layers.LeakyReLU(alpha=0.3),
        layers.Reshape((7, 7, 64)),
        layers.Conv2DTranspose(
            64, (3, 3), strides=(2, 2), padding="same", kernel_initializer="he_normal"
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.Conv2DTranspose(
            32, (3, 3), strides=(2, 2), padding="same", kernel_initializer="he_normal"
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.Conv2D(1, (3, 3), padding="same", activation="sigmoid"),
    ]
)


# Load the MNIST dataset "Use 10000 training images and 2000 test images from the dataset."
(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train[:10000]", "test[:2000]"],
    shuffle_files=True,
    as_supervised=False,
    with_info=True,
)


# Normalize images only
def normalize_img(data):
    image = tf.cast(data["image"], tf.float32) / 255.0
    image = tf.reshape(image, (28, 28, 1))
    return image


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

# Precompute clean images for training and testing
train_images = list(ds_train.as_numpy_iterator())
train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)

test_images = list(ds_test.as_numpy_iterator())
test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)

print(
    f"Loaded {train_images.shape[0]} training images and {test_images.shape[0]} test images"
)

# Noise parameters
noise_stddev = 0.3


# Function to generate noisy dataset for an epoch
def create_noisy_dataset(clean_images, batch_size=128, noise_std=0.3):
    """Create a dataset with fresh noise added to clean images"""
    # Add Gaussian noise
    noise = tf.random.normal(shape=tf.shape(clean_images), mean=0.0, stddev=noise_std)
    noisy_images = clean_images + noise
    noisy_images = tf.clip_by_value(noisy_images, 0.0, 1.0)

    # Create dataset with (noisy, clean) pairs
    dataset = tf.data.Dataset.from_tensor_slices((noisy_images, clean_images))
    dataset = dataset.shuffle(buffer_size=len(clean_images))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"], jit_compile=True)

# Display model architecture
model.summary()

# Custom training loop to regenerate noise each epoch
epochs = 25
history = {
    "loss": [],
    "mae": [],
    "mse": [],
    "val_loss": [],
    "val_mae": [],
    "val_mse": [],
}

print("\nStarting training with unique noise per epoch...")
print("=" * 50)

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    # Generate new noisy training dataset for this epoch
    ds_train_epoch = create_noisy_dataset(
        train_images, batch_size=128, noise_std=noise_stddev
    )

    # Generate new noisy validation dataset for this epoch
    ds_test_epoch = create_noisy_dataset(
        test_images, batch_size=128, noise_std=noise_stddev
    )

    # Train for one epoch
    epoch_history = model.fit(
        ds_train_epoch, epochs=1, validation_data=ds_test_epoch, verbose=1
    )

    # Store metrics
    history["loss"].append(epoch_history.history["loss"][0])
    history["mae"].append(epoch_history.history["mae"][0])
    history["mse"].append(epoch_history.history["mse"][0])
    history["val_loss"].append(epoch_history.history["val_loss"][0])
    history["val_mae"].append(epoch_history.history["val_mae"][0])
    history["val_mse"].append(epoch_history.history["val_mse"][0])

print("\n" + "=" * 50)
print("Training completed!")

# Evaluate the model on a fresh noisy test set
print("\n" + "=" * 50)
print("Final Evaluation on Test Set (with fresh noise):")
print("=" * 50)
ds_test_final = create_noisy_dataset(
    test_images, batch_size=128, noise_std=noise_stddev
)
results = model.evaluate(ds_test_final, verbose=1)
print(f"\nTest Loss (MSE): {results[0]:.6f}")
print(f"Test MAE: {results[1]:.6f}")
print(f"Test MSE: {results[2]:.6f}")


model.save("mnist_autoencoder_denoiser_model.keras")
print("\n✓ Full model saved to: mnist_autoencoder_denoiser_model.keras")
