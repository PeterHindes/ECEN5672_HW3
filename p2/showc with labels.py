import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

print("Loading saved denoising model...")
model = keras.models.load_model("mnist_autoencoder_denoiser_model_with_labels.keras")
print("✓ Model loaded successfully!\n")

print("Loading test data...")
ds_test = tfds.load(
    "mnist",
    split="test[:2000]",
    shuffle_files=True,
    as_supervised=False,
)


def normalize_img(data):
    image = tf.cast(data["image"], tf.float32) / 255.0
    label = data["label"]
    return image, label


ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
print("✓ Test data loaded\n")

print("Press Ctrl+C to exit\n")

try:
    while True:
        # Get a random image and its label
        print("Selecting random image...")
        for images, labels in ds_test.take(1):
            random_idx = np.random.randint(0, images.shape[0])
            clean_image = images[random_idx : random_idx + 1]
            true_label = labels[random_idx]
            break

        # Add noise to the image (same as b.py)
        noise = tf.random.normal(shape=tf.shape(clean_image), mean=0.0, stddev=0.3)
        noisy_image = clean_image + noise

        print("Generating denoised reconstruction and predicting label...")
        denoised_image, predicted_label_probs = model.predict(noisy_image, verbose=0)
        predicted_label = np.argmax(predicted_label_probs[0])

        # Calculate reconstruction error (comparing to clean image)
        mse = np.mean((clean_image.numpy() - denoised_image) ** 2)
        mae = np.mean(np.abs(clean_image.numpy() - denoised_image))

        # Calculate noise level
        noise_mse = np.mean((clean_image.numpy() - noisy_image.numpy()) ** 2)

        print(f"\nReconstruction Metrics:")
        print(f"  Noise MSE:   {noise_mse:.6f}")
        print(f"  Denoised MSE:  {mse:.6f}")
        print(f"  Denoised MAE:  {mae:.6f}")
        print(f"  Improvement:   {((noise_mse - mse) / noise_mse * 100):.2f}%")
        print(f"\nLabel Prediction:")
        print(f"  True Label:      {true_label.numpy()}")
        print(f"  Predicted Label: {predicted_label}\n")

        # Create 3-way comparison: Clean | Noisy | Denoised
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(clean_image[0].numpy().squeeze(), cmap="gray")
        axes[0].set_title(
            f"Clean Original\n(Label: {true_label.numpy()})",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].axis("off")

        axes[1].imshow(noisy_image[0].numpy().squeeze(), cmap="gray")
        axes[1].set_title(
            f"Noisy Input\n(MSE: {noise_mse:.4f})", fontsize=14, fontweight="bold"
        )
        axes[1].axis("off")

        axes[2].imshow(denoised_image[0].squeeze(), cmap="gray")
        axes[2].set_title(
            f"Denoised Output\n(Predicted: {predicted_label}, MSE: {mse:.4f})",
            fontsize=14,
            fontweight="bold",
        )
        axes[2].axis("off")

        improvement = (noise_mse - mse) / noise_mse * 100
        plt.suptitle(
            f"Denoising Autoencoder\nNoise Reduction: {improvement:.2f}%",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        print("Opening visualization window...")
        plt.show()

except KeyboardInterrupt:
    print("\n\nExiting...")
