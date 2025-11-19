import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

print("Loading saved label-only denoising model...")
model = keras.models.load_model("mnist_autoencoder_label_only.keras")
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

        # Add noise to the image
        noise = tf.random.normal(shape=tf.shape(clean_image), mean=0.0, stddev=0.3)
        noisy_image = clean_image + noise

        print("Generating denoised reconstruction and predictions...")
        # Model now has only 2 outputs: reconstruction and digit classification
        denoised_image, digit_probs = model.predict(noisy_image, verbose=0)

        predicted_digit = np.argmax(digit_probs[0])

        # Get confidence scores
        digit_confidence = digit_probs[0][predicted_digit]

        # Calculate reconstruction error (comparing to clean image)
        mse = np.mean((clean_image.numpy() - denoised_image) ** 2)
        mae = np.mean(np.abs(clean_image.numpy() - denoised_image))

        # Calculate noise level
        noise_mse = np.mean((clean_image.numpy() - noisy_image.numpy()) ** 2)

        print(f"\nReconstruction Metrics:")
        print(f"  Noise MSE:     {noise_mse:.6f}")
        print(f"  Denoised MSE:  {mse:.6f}")
        print(f"  Denoised MAE:  {mae:.6f}")
        print(f"  Improvement:   {((noise_mse - mse) / noise_mse * 100):.2f}%")
        print(f"\nDigit Classification:")
        print(f"  True Label:       {true_label.numpy()}")
        print(f"  Predicted Digit:  {predicted_digit}")
        print(f"  Confidence:       {digit_confidence:.2%}")
        print(
            f"  Correct:          {'✓' if true_label.numpy() == predicted_digit else '✗'}"
        )

        # Display top 3 digit probabilities
        print(f"\nTop 3 Digit Predictions:")
        top_3_digits = np.argsort(digit_probs[0])[-3:][::-1]
        for rank, digit_idx in enumerate(top_3_digits, 1):
            prob = digit_probs[0][digit_idx]
            marker = "★" if digit_idx == true_label.numpy() else " "
            print(f"  {rank}. Digit {digit_idx}: {prob:.2%} {marker}")

        # Create 2-row visualization
        # Top row: Clean | Noisy | Denoised
        # Bottom row: Digit Probabilities (full width)
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.3)

        # Top Row - Images
        # Panel 1: Clean Original
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(clean_image[0].numpy().squeeze(), cmap="gray")
        ax0.set_title(
            f"Clean Original\n(True Label: {true_label.numpy()})",
            fontsize=14,
            fontweight="bold",
        )
        ax0.axis("off")

        # Panel 2: Noisy Input
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(noisy_image[0].numpy().squeeze(), cmap="gray")
        ax1.set_title(
            f"Noisy Input\n(MSE: {noise_mse:.4f})", fontsize=14, fontweight="bold"
        )
        ax1.axis("off")

        # Panel 3: Denoised Output
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(denoised_image[0].squeeze(), cmap="gray")
        ax2.set_title(
            f"Denoised Output\n(Predicted: {predicted_digit}, MSE: {mse:.4f})",
            fontsize=14,
            fontweight="bold",
        )
        ax2.axis("off")

        # Bottom Row - Probability Distributions
        # Digit probabilities (full width)
        digit_subplot = fig.add_subplot(gs[1, :])
        digit_bars = digit_subplot.bar(
            range(10), digit_probs[0], color="steelblue", alpha=0.7
        )
        digit_bars[predicted_digit].set_color("darkgreen")
        digit_bars[predicted_digit].set_alpha(1.0)
        if true_label.numpy() != predicted_digit:
            digit_bars[true_label.numpy()].set_color("darkred")
            digit_bars[true_label.numpy()].set_alpha(0.8)
        digit_subplot.set_xlabel("Digit", fontsize=11)
        digit_subplot.set_ylabel("Probability", fontsize=11)
        digit_subplot.set_title(
            f"Digit Classification Probabilities (Predicted: {predicted_digit}, Confidence: {digit_confidence:.1%})",
            fontsize=12,
            fontweight="bold",
        )
        digit_subplot.set_xticks(range(10))
        digit_subplot.set_ylim([0, 1])
        digit_subplot.grid(axis="y", alpha=0.3)

        improvement = (noise_mse - mse) / noise_mse * 100
        correct_prediction = (
            "✓ Correct" if true_label.numpy() == predicted_digit else "✗ Incorrect"
        )

        plt.suptitle(
            f"Label-Only Denoising Autoencoder ({correct_prediction})\n"
            f"Noise Reduction: {improvement:.2f}% | "
            f"Bottleneck: Digit={predicted_digit} (conf: {digit_confidence:.1%}) + 5-dim unsupervised style encoding",
            fontsize=16,
            fontweight="bold",
        )

        print("\nOpening visualization window...")
        plt.show()

except KeyboardInterrupt:
    print("\n\nExiting...")
