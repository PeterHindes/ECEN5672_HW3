import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

print("Loading saved EMNIST lowercase-only denoising model...")
model = keras.models.load_model("emnist_lowercase_autoencoder.keras")
print("✓ Model loaded successfully!\n")

print("Loading test data...")
ds_test = tfds.load(
    "emnist/byclass",
    split="test",
    shuffle_files=True,
    as_supervised=True,
)


# Filter for lowercase letters only (labels 36-61 in ByClass)
def filter_lowercase(image, label):
    """Keep only lowercase letters (labels 36-61)"""
    return tf.logical_and(label >= 36, label <= 61)


ds_test = ds_test.filter(filter_lowercase)


# Lowercase letter mapping (labels 0-25 correspond to a-z)
def label_to_char(label):
    """Convert remapped label (0-25) to lowercase character"""
    return chr(ord("a") + label)


def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    # Remap label from 36-61 to 0-25
    remapped_label = label - 36
    return image, remapped_label


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
            true_label = labels[random_idx].numpy()
            break

        true_char = label_to_char(true_label)

        # Add noise to the image
        noise = tf.random.normal(shape=tf.shape(clean_image), mean=0.0, stddev=0.3)
        noisy_image = clean_image + noise

        print("Generating denoised reconstruction and predictions...")
        denoised_image, char_probs = model.predict(noisy_image, verbose=0)

        predicted_label = np.argmax(char_probs[0])
        predicted_char = label_to_char(predicted_label)

        # Get confidence scores
        char_confidence = char_probs[0][predicted_label]

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
        print(f"\nCharacter Classification:")
        print(f"  True Character:      '{true_char}' (label {true_label})")
        print(f"  Predicted Character: '{predicted_char}' (label {predicted_label})")
        print(f"  Confidence:          {char_confidence:.2%}")
        print(f"  Correct:             {'✓' if true_label == predicted_label else '✗'}")

        # Display top 5 character probabilities
        print(f"\nTop 5 Character Predictions:")
        top_5_chars = np.argsort(char_probs[0])[-5:][::-1]
        for rank, char_idx in enumerate(top_5_chars, 1):
            prob = char_probs[0][char_idx]
            char = label_to_char(char_idx)
            marker = "★" if char_idx == true_label else " "
            print(f"  {rank}. '{char}': {prob:.2%} {marker}")

        # Create 2-row visualization
        # Top row: Clean | Noisy | Denoised
        # Bottom row: Character Probabilities (full width)
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.3)

        # Top Row - Images
        # Panel 1: Clean Original
        ax0 = fig.add_subplot(gs[0, 0])
        # Transpose for correct display (EMNIST images are stored rotated)
        ax0.imshow(clean_image[0].numpy().squeeze().T, cmap="gray")
        ax0.set_title(
            f"Clean Original\n(True: '{true_char}')",
            fontsize=14,
            fontweight="bold",
        )
        ax0.axis("off")

        # Panel 2: Noisy Input
        ax1 = fig.add_subplot(gs[0, 1])
        # Transpose for correct display (EMNIST images are stored rotated)
        ax1.imshow(noisy_image[0].numpy().squeeze().T, cmap="gray")
        ax1.set_title(
            f"Noisy Input\n(MSE: {noise_mse:.4f})", fontsize=14, fontweight="bold"
        )
        ax1.axis("off")

        # Panel 3: Denoised Output
        ax2 = fig.add_subplot(gs[0, 2])
        # Transpose for correct display (EMNIST images are stored rotated)
        ax2.imshow(denoised_image[0].squeeze().T, cmap="gray")
        ax2.set_title(
            f"Denoised Output\n(Predicted: '{predicted_char}', MSE: {mse:.4f})",
            fontsize=14,
            fontweight="bold",
        )
        ax2.axis("off")

        # Bottom Row - Probability Distributions
        # Character probabilities (full width)
        char_subplot = fig.add_subplot(gs[1, :])

        # Show all 26 lowercase letter probabilities
        char_labels = [chr(ord("a") + i) for i in range(26)]
        char_bars = char_subplot.bar(
            range(26), char_probs[0], color="steelblue", alpha=0.7
        )

        # Highlight predicted and true
        char_bars[predicted_label].set_color("darkgreen")
        char_bars[predicted_label].set_alpha(1.0)
        if true_label != predicted_label:
            char_bars[true_label].set_color("darkred")
            char_bars[true_label].set_alpha(0.8)

        char_subplot.set_xticks(range(26))
        char_subplot.set_xticklabels([f"'{c}'" for c in char_labels], fontsize=10)
        char_subplot.set_xlabel("Lowercase Letter", fontsize=11)
        char_subplot.set_ylabel("Probability", fontsize=11)
        char_subplot.set_title(
            f"Character Classification Probabilities (Predicted: '{predicted_char}', Confidence: {char_confidence:.1%})",
            fontsize=12,
            fontweight="bold",
        )
        char_subplot.set_ylim([0, 1])
        char_subplot.grid(axis="y", alpha=0.3)

        improvement = (noise_mse - mse) / noise_mse * 100
        correct_prediction = (
            "✓ Correct" if true_label == predicted_label else "✗ Incorrect"
        )

        plt.suptitle(
            f"EMNIST Lowercase-Only Denoising Autoencoder ({correct_prediction})\n"
            f"Noise Reduction: {improvement:.2f}% | "
            f"True: '{true_char}' → Predicted: '{predicted_char}' (conf: {char_confidence:.1%}) + 12-dim unsupervised style encoding",
            fontsize=16,
            fontweight="bold",
        )

        print("\nOpening visualization window...")
        plt.show()

except KeyboardInterrupt:
    print("\n\nExiting...")
