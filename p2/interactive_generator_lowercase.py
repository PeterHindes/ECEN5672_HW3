import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.widgets import RadioButtons, Slider
from tensorflow import keras

print("Loading saved lowercase-only model...")
model = keras.models.load_model("emnist_lowercase_autoencoder.keras")
print("✓ Model loaded successfully!\n")

# Get the decoder part of the model
# We'll feed it manual character and style values
decoder_input_layer = model.get_layer("combined_bottleneck")
decoder_output = model.get_layer("reconstructed_image").output

# Create a model that takes the combined bottleneck as input
decoder_model = keras.Model(inputs=decoder_input_layer.output, outputs=decoder_output)

# Initial values
current_letter = 0  # 'a'
current_style_values = np.zeros(
    12, dtype=np.float32
)  # 12-dim style, all zeros initially

# Create figure and layout
fig = plt.figure(figsize=(16, 9))

# Title at top
fig.text(
    0.5,
    0.96,
    "Interactive Lowercase Letter Generator",
    ha="center",
    fontsize=18,
    fontweight="bold",
)

# Image display (top center, larger)
ax_image = plt.axes([0.35, 0.50, 0.40, 0.40])
ax_image.set_title("Generated Letter", fontsize=14, fontweight="bold")
ax_image.axis("off")

# Radio buttons for letter selection - ALL IN ONE COLUMN
ax_radio_all = plt.axes([0.05, 0.30, 0.15, 0.55])
ax_radio_all.set_title("Select Letter (a-z)", fontsize=11, fontweight="bold")
radio_all = RadioButtons(
    ax_radio_all,
    labels=[
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ],
    active=0,
)

# Sliders for style dimensions (bottom half)
# Create 12 slider axes in 4 columns x 3 rows
sliders = []
slider_height = 0.03
slider_width = 0.17

fig.text(
    0.58,
    0.38,
    "Style Controls (S0-S11, range: -1.0 to 1.0)",
    ha="center",
    fontsize=11,
    fontweight="bold",
)

# Positions for sliders in grid (4 columns x 3 rows)
for i in range(12):
    col = i % 4  # columns 0, 1, 2, 3
    row = i // 4  # rows 0, 1, 2

    # Calculate position
    left = 0.28 + col * 0.18
    bottom = 0.28 - row * 0.10

    ax_slider = plt.axes([left, bottom, slider_width, slider_height])
    slider = Slider(
        ax_slider,
        f"S{i}",
        -1.0,
        1.0,
        valinit=0.0,
        valstep=0.05,
        color="steelblue",
    )
    sliders.append(slider)


def generate_image():
    """Generate image from current letter and style values"""
    # Create one-hot encoded character
    char_bottleneck = np.zeros((1, 26), dtype=np.float32)
    char_bottleneck[0, current_letter] = 1.0

    # Style values (tanh activation, range -1 to 1)
    style_bottleneck = current_style_values.reshape(1, -1)

    # Combine character and style
    combined = np.concatenate([char_bottleneck, style_bottleneck], axis=1)

    # Generate image
    generated_image = decoder_model.predict(combined, verbose=0)

    return generated_image[0, :, :, 0]


def update_display():
    """Update the displayed image"""
    generated_image = generate_image()

    ax_image.clear()
    # Transpose for correct EMNIST orientation
    ax_image.imshow(generated_image.T, cmap="gray")

    letter_char = chr(ord("a") + current_letter)
    style_str = "[" + ", ".join([f"{v:.2f}" for v in current_style_values]) + "]"

    ax_image.set_title(
        f"Generated Letter: '{letter_char}'\n"
        f"Style: {style_str[:60]}{'...' if len(style_str) > 60 else ''}",
        fontsize=14,
        fontweight="bold",
    )
    ax_image.axis("off")

    plt.draw()


def on_radio_change(label):
    """Handle letter selection"""
    global current_letter
    current_letter = ord(label) - ord("a")
    update_display()


def on_slider_change(val):
    """Handle slider changes"""
    global current_style_values
    for i, slider in enumerate(sliders):
        current_style_values[i] = slider.val
    update_display()


def reset_style():
    """Reset all style sliders to zero"""
    global current_style_values
    current_style_values = np.zeros(12, dtype=np.float32)
    for slider in sliders:
        slider.set_val(0.0)
    update_display()


def randomize_style():
    """Randomize all style sliders"""
    global current_style_values
    current_style_values = np.random.uniform(-1.0, 1.0, size=12).astype(np.float32)
    for i in range(12):
        sliders[i].set_val(current_style_values[i])
    update_display()  # Explicitly update after all sliders are set


# Connect callbacks
radio_all.on_clicked(on_radio_change)
for slider in sliders:
    slider.on_changed(on_slider_change)

# Add control buttons
from matplotlib.widgets import Button

ax_reset = plt.axes([0.35, 0.02, 0.12, 0.035])
btn_reset = Button(ax_reset, "Reset Style", color="lightcoral", hovercolor="coral")
btn_reset.on_clicked(lambda event: reset_style())

ax_random = plt.axes([0.52, 0.02, 0.12, 0.035])
btn_random = Button(ax_random, "Random Style", color="lightgreen", hovercolor="green")
btn_random.on_clicked(lambda event: randomize_style())

# Initial display
update_display()

print("\n" + "=" * 70)
print("Interactive Lowercase Letter Generator Ready!")
print("=" * 70)
print("\nInstructions:")
print("  1. Click radio buttons to select which letter (a-z)")
print("  2. Adjust 12 style sliders (S0-S11) to control appearance")
print("  3. Each style dimension affects different aspects of the letter")
print("  4. Use 'Reset Style' for neutral (zero) style")
print("  5. Use 'Random Style' to explore random variations")
print("  6. The image updates in real-time!")
print("\nExperiment:")
print("  - Try different style combinations for the same letter")
print("  - See how each style dimension affects the rendering")
print("  - Compare how different letters respond to the same style")
print("  - Notice which dimensions have the strongest effects")
print("\nClose the window to exit.")
print("=" * 70)

plt.show()
