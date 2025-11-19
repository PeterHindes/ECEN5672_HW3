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

# Title at top
fig.text(
    0.5,
    0.95,
    "Interactive Digit Generator",
    ha="center",
    fontsize=18,
    fontweight="bold",
)

# Image display (top center)
ax_image = plt.axes([0.35, 0.45, 0.35, 0.45])
ax_image.set_title("Generated Digit", fontsize=14, fontweight="bold")
ax_image.axis("off")

# Radio buttons for digit selection (left side)
ax_radio = plt.axes([0.05, 0.30, 0.15, 0.55])
ax_radio.set_title("Select Digit", fontsize=12, fontweight="bold")
radio = RadioButtons(
    ax_radio,
    labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    active=0,
)

# Title for sliders
fig.text(
    0.65,
    0.38,
    "Style Controls (V0-V4)",
    ha="center",
    fontsize=11,
    fontweight="bold",
)

# Create 5 slider axes (right side, vertically stacked)
sliders = []
slider_height = 0.03
slider_width = 0.25

for i in range(5):
    left = 0.52
    bottom = 0.30 - i * 0.06

    ax_slider = plt.axes([left, bottom, slider_width, slider_height])
    slider = Slider(
        ax_slider,
        f"V{i}",
        0.0,
        1.0,
        valinit=current_variant_values[i],
        valstep=0.01,
    )
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
    for i in range(5):
        current_variant_values[i] = sliders[i].val
    update_display()


def reset_style():
    """Reset all variant sliders to equal values"""
    global current_variant_values
    current_variant_values = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    for i in range(5):
        sliders[i].set_val(0.2)


def randomize_style():
    """Randomize all variant sliders"""
    global current_variant_values
    current_variant_values = np.random.dirichlet(np.ones(5)).astype(np.float32)
    for i in range(5):
        sliders[i].set_val(current_variant_values[i])
    update_display()  # Explicitly update after all sliders are set


# Connect callbacks
radio.on_clicked(on_radio_change)
for slider in sliders:
    slider.on_changed(on_slider_change)

# Add control buttons
from matplotlib.widgets import Button

ax_reset = plt.axes([0.30, 0.02, 0.12, 0.04])
btn_reset = Button(ax_reset, "Reset Style", color="lightcoral", hovercolor="coral")
btn_reset.on_clicked(lambda event: reset_style())

ax_random = plt.axes([0.52, 0.02, 0.12, 0.04])
btn_random = Button(ax_random, "Random Style", color="lightgreen", hovercolor="green")
btn_random.on_clicked(lambda event: randomize_style())

# Initial display
update_display()

print("\n" + "=" * 70)
print("Interactive Digit Generator Ready!")
print("=" * 70)
print("Controls: Select digit (left), adjust 5 style sliders (V0-V4)")
print("Use Reset/Random buttons to control style")
print("=" * 70)

plt.show()
