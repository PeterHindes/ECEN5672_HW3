import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.widgets import RadioButtons, Slider
from tensorflow import keras

print("Loading saved label-only model...")
model = keras.models.load_model("mnist_autoencoder_label_only.keras")
print("✓ Model loaded successfully!\n")

# Get the decoder part of the model
# We'll feed it manual digit and variant values
decoder_input_layer = model.get_layer("combined_bottleneck")
decoder_output = model.get_layer("reconstructed_image").output

# Create a model that takes the combined bottleneck as input
decoder_model = keras.Model(inputs=decoder_input_layer.output, outputs=decoder_output)

# Initial values
current_digit = 0
current_variant_values = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal distribution

# Create figure and layout
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(3, 2, height_ratios=[2, 0.3, 0.3], hspace=0.4, wspace=0.4)

# Image display (top)
ax_image = fig.add_subplot(gs[0, :])
ax_image.set_title("Generated Digit", fontsize=16, fontweight="bold")
ax_image.axis("off")

# Radio buttons for digit selection (bottom left)
ax_radio = fig.add_subplot(gs[1:, 0])
ax_radio.set_title("Select Digit (0-9)", fontsize=12, fontweight="bold")
radio = RadioButtons(
    ax_radio,
    labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    active=0,
)

# Sliders for variant probabilities (bottom right)
ax_sliders = fig.add_subplot(gs[1:, 1])
ax_sliders.axis("off")

# Create 5 slider axes
slider_axes = []
sliders = []
slider_height = 0.08
slider_spacing = 0.12
slider_bottom_start = 0.25

for i in range(5):
    ax_slider = plt.axes(
        [0.60, slider_bottom_start + i * slider_spacing, 0.30, slider_height]
    )
    slider = Slider(
        ax_slider,
        f"Variant {i}",
        0.0,
        1.0,
        valinit=current_variant_values[i],
        valstep=0.01,
    )
    slider_axes.append(ax_slider)
    sliders.append(slider)


def generate_image():
    """Generate image from current digit and variant values"""
    # Create one-hot encoded digit
    digit_bottleneck = np.zeros((1, 10), dtype=np.float32)
    digit_bottleneck[0, current_digit] = 1.0

    # Normalize variant values to sum to 1 (softmax-like)
    variant_values = np.array(current_variant_values, dtype=np.float32).reshape(1, -1)
    variant_sum = np.sum(variant_values)
    if variant_sum > 0:
        variant_bottleneck = variant_values / variant_sum
    else:
        variant_bottleneck = np.ones((1, 5), dtype=np.float32) / 5.0

    # Combine digit and variant
    combined = np.concatenate([digit_bottleneck, variant_bottleneck], axis=1)

    # Generate image
    generated_image = decoder_model.predict(combined, verbose=0)

    return generated_image[0, :, :, 0], variant_bottleneck[0]


def update_display():
    """Update the displayed image"""
    generated_image, normalized_variants = generate_image()

    ax_image.clear()
    ax_image.imshow(generated_image, cmap="gray")
    ax_image.set_title(
        f"Generated Digit: {current_digit}\n"
        f"Variant Distribution: [{', '.join([f'{v:.2f}' for v in normalized_variants])}]",
        fontsize=14,
        fontweight="bold",
    )
    ax_image.axis("off")

    # Update slider colors to show which is dominant
    max_variant_idx = np.argmax(normalized_variants)
    for i, slider in enumerate(sliders):
        if i == max_variant_idx:
            slider.label.set_color("darkviolet")
            slider.label.set_weight("bold")
        else:
            slider.label.set_color("black")
            slider.label.set_weight("normal")

    plt.draw()


def on_radio_change(label):
    """Handle digit selection"""
    global current_digit
    current_digit = int(label)
    update_display()


def on_slider_change(val):
    """Handle slider changes"""
    global current_variant_values
    for i, slider in enumerate(sliders):
        current_variant_values[i] = slider.val
    update_display()


# Connect callbacks
radio.on_clicked(on_radio_change)
for slider in sliders:
    slider.on_changed(on_slider_change)

# Add instructions
instructions_text = (
    "Interactive Digit Generator\n\n"
    "Controls:\n"
    "• Select digit using radio buttons (left)\n"
    "• Adjust variant sliders (right) to change style\n"
    "• Variant values are auto-normalized to sum to 1.0\n"
    "• Dominant variant shown in purple\n\n"
    "The model generates images using ONLY:\n"
    "  - Which digit (0-9)\n"
    "  - Variant distribution (5 values)"
)

fig.text(
    0.5,
    0.02,
    instructions_text,
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)

# Initial display
update_display()

print("\n" + "=" * 70)
print("Interactive Generator Ready!")
print("=" * 70)
print("\nInstructions:")
print("  1. Click radio buttons to select which digit (0-9)")
print("  2. Adjust sliders to control the 5 variant probabilities")
print("  3. The image updates in real-time!")
print("\nExperiment:")
print("  - Try different variant combinations for the same digit")
print("  - See how variants affect writing style")
print("  - Notice which variants have the strongest effect")
print("\nClose the window to exit.")
print("=" * 70)

plt.show()
