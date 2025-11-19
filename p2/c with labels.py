import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

# print("TensorFlow Version:", tf.__version__)
# print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# Define the CNN model using the Functional API
input_img = keras.Input(shape=(28, 28, 1), name="input_image")

# Encoder
x = layers.Conv2D(16, (3, 3), padding="same")(input_img)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.MaxPooling2D((2, 2))(x)  # 28→14
x = layers.Conv2D(32, (3, 3), padding="same")(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.MaxPooling2D((2, 2))(x)  # 14→7
x = layers.Conv2D(16, (3, 3), padding="same")(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.MaxPooling2D((2, 2))(x)  # 7→3
x = layers.Conv2D(32, (3, 3), padding="same")(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Flatten()(x)  # 3×3×64 = 576
x = layers.Dense(128)(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Dropout(0.3)(x)
encoder_output = layers.Dense(12, name="encoder_bottleneck_output")(x)
encoder_output = layers.LeakyReLU(alpha=0.2)(encoder_output)

# Classification head - larger to reduce constraint on bottleneck
classification_head = layers.Dense(3136)(encoder_output)
classification_head = layers.LeakyReLU(alpha=0.2)(classification_head)
classification_head = layers.Dropout(0.1)(classification_head)
classification_head = layers.Dense(128)(classification_head)
classification_head = layers.LeakyReLU(alpha=0.2)(classification_head)
classification_head = layers.Dropout(0.1)(classification_head)
classification_head = layers.Dense(32)(classification_head)
classification_head = layers.LeakyReLU(alpha=0.2)(classification_head)
classification_output = layers.Dense(10, activation="softmax", name="label_output")(
    classification_head
)

# Decoder
decoder_input = layers.Dense(3136)(encoder_output)
decoder_input = layers.LeakyReLU(alpha=0.2)(decoder_input)
decoder_input = layers.Reshape((7, 7, 64))(decoder_input)
x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(
    decoder_input
)  # 7→14
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(x)  # 14→28
x = layers.LeakyReLU(alpha=0.2)(x)
reconstructed_output = layers.Conv2D(
    1, (3, 3), padding="same", activation="sigmoid", name="reconstructed_image"
)(x)

# Create the model with two outputs
model = keras.Model(
    inputs=input_img, outputs=[reconstructed_output, classification_output]
)


# Load the MNIST dataset "Use 10000 training images and 2000 test images from the dataset."
(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train[:10000]", "test[:2000]"],
    shuffle_files=True,
    as_supervised=False,
    with_info=True,
)


# add normalize and add noise to the input images while keeping the target image and label unchanged
def normalize_img(data):
    image = tf.cast(data["image"], tf.float32) / 255.0
    label = data["label"]
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.3)
    noisy_image = image + noise
    return noisy_image, {"reconstructed_image": image, "label_output": label}


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Compile the model
# Note: label_output has small weight (0.1) to encourage encoder to learn meaningful representations
# while still prioritizing reconstruction (0.9) - this helps the bottleneck encode digit identity
model.compile(
    optimizer="adam",
    loss={
        "reconstructed_image": "mse",
        "label_output": "sparse_categorical_crossentropy",
    },
    loss_weights={"reconstructed_image": 0.9, "label_output": 0.1},
    metrics={"reconstructed_image": ["mae", "mse"], "label_output": "accuracy"},
    jit_compile=True,
)

# Display model architecture
model.summary()

# Train the model
history = model.fit(
    ds_train,
    epochs=25,
    validation_data=ds_test,
)

# Evaluate the model
print("\n" + "=" * 50)
print("Final Evaluation on Test Set:")
print("=" * 50)
results = model.evaluate(ds_test, verbose=1)
print(f"\nTotal Test Loss: {results[0]:.6f}")
print(f"Reconstruction Test Loss (MSE): {results[1]:.6f}")
print(f"Label Prediction Test Loss: {results[2]:.6f}")
print(f"Reconstruction Test MAE: {results[3]:.6f}")
print(f"Reconstruction Test MSE: {results[4]:.6f}")
print(f"Label Prediction Test Accuracy: {results[5]:.6f}")


model.save("mnist_autoencoder_denoiser_model_with_labels.keras")
print("\n✓ Full model saved to: mnist_autoencoder_denoiser_model_with_labels.keras")
