import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.widgets import Button, RadioButtons, Slider
from tensorflow import keras

print("Loading saved adversarial lowercase model...")
model = keras.models.load_model("emnist_lowercase_adversarial.keras")
print("✓ Model loaded successfully!\n")

# Get the decoder part of the model
# We'll feed it manual character and style values
decoder_input_layer = model.get_layer("combined_bottleneck")
decoder_output = model.get_layer("reconstructed_image").output

# Create a model that takes the combined bottleneck as input
decoder_model = keras.Model(inputs=decoder_input_layer.output, outputs=decoder_output)

# Also create a style predictor model (takes only style bottleneck)
# Create a style predictor model that takes style bottleneck as input
# We'll extract this from the trained model's computation graph
try:
    # Get the style encoding layer output
    style_bottleneck_layer = model.get_layer("style_encoding")
    style_predictor_layer = model.get_layer("style_predictor")

    # Create a new model: style bottleneck input -> style predictor output
    # We need to trace the path from style_encoding to style_predictor
    style_input = keras.Input(shape=(12,), name="manual_style_input")

    # Find all layers between style_encoding and style_predictor
    # by examining the model's layer connections
    # The style predictor path should be: style_encoding -> dense -> leaky_relu -> dropout -> dense (style_predictor)

    # Simple approach: create functional model from existing layers
    # Get intermediate layers by checking inbound nodes of style_predictor
    x = style_input

    # Find layers that connect style_encoding to style_predictor
    connected_layers = []
    for layer in model.layers:
        # Check if layer is in the style predictor path
        if hasattr(layer, "inbound_nodes") and len(layer.inbound_nodes) > 0:
            for node in layer.inbound_nodes:
                if hasattr(node, "inbound_layers"):
                    inbound_names = [l.name for l in node.inbound_layers]
                    if "style_encoding" in inbound_names or any(
                        "style" in name for name in inbound_names
                    ):
                        connected_layers.append(layer)

    # Build the predictor from style input
    # Recreate the architecture: Dense(64) -> LeakyReLU -> Dropout -> Dense(26)
    x = keras.layers.Dense(64, name="sp_dense_64")(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(26, activation="softmax", name="sp_dense_26")(x)

    manual_style_predictor = keras.Model(inputs=style_input, outputs=x)

    # Try to copy weights from the main model
    # Find the dense layers in the style predictor path
    dense_layers = [
        l
        for l in model.layers
        if isinstance(l, keras.layers.Dense) and l.output.shape[-1] in [64, 26]
    ]

    # Copy weights to our manual predictor
    manual_dense_layers = [
        l for l in manual_style_predictor.layers if isinstance(l, keras.layers.Dense)
    ]

    if len(dense_layers) >= 2 and len(manual_dense_layers) >= 2:
        # Find the 64-dim and 26-dim dense layers
        for orig_layer in dense_layers[
            -2:
        ]:  # Last 2 dense layers likely style predictor
            if orig_layer.output.shape[-1] == 64:
                manual_dense_layers[0].set_weights(orig_layer.get_weights())
            elif orig_layer.output.shape[-1] == 26:
                manual_dense_layers[1].set_weights(orig_layer.get_weights())

    print("✓ Style predictor extracted from model")

except Exception as e:
    print(f"⚠ Could not extract style predictor: {e}")
    print("  Creating random style predictor for demonstration")

    # Fallback: create a simple predictor with random weights
    style_input = keras.Input(shape=(12,))
    x = keras.layers.Dense(64)(style_input)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(26, activation="softmax")(x)
    manual_style_predictor = keras.Model(inputs=style_input, outputs=x)

# Initial values
current_letter = 0  # 'a'
current_style_values = np.zeros(12, dtype=np.float32)

# Create figure and layout
fig = plt.figure(figsize=(18, 10))

# Title at top
fig.text(
    0.5,
    0.97,
    "Interactive Adversarial Lowercase Generator",
    ha="center",
    fontsize=20,
    fontweight="bold",
)
fig.text(
    0.5,
    0.94,
    "With Style Disentanglement Feedback",
    ha="center",
    fontsize=12,
    style="italic",
    color="gray",
)

# Main image display (top left, larger)
ax_image = plt.axes([0.30, 0.50, 0.35, 0.38])
ax_image.set_title("Generated Letter", fontsize=14, fontweight="bold")
ax_image.axis("off")

# Style predictor feedback panel (top right)
ax_feedback = plt.axes([0.68, 0.50, 0.28, 0.38])
ax_feedback.set_title("Style Disentanglement Analysis", fontsize=12, fontweight="bold")
ax_feedback.axis("off")

# Radio buttons for letter selection
ax_radio_all = plt.axes([0.03, 0.30, 0.15, 0.60])
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
    col = i % 4
    row = i // 4

    left = 0.25 + col * 0.18
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


def get_style_prediction():
    """Get style predictor's guess from style values alone"""
    style_bottleneck = current_style_values.reshape(1, -1)

    try:
        prediction = manual_style_predictor.predict(style_bottleneck, verbose=0)
        predicted_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_idx]

        # Get top 3 predictions
        top3_indices = np.argsort(prediction[0])[-3:][::-1]
        top3_chars = [chr(ord("a") + idx) for idx in top3_indices]
        top3_confs = [prediction[0][idx] for idx in top3_indices]

        return predicted_idx, confidence, top3_chars, top3_confs
    except Exception as e:
        # Fallback if style predictor not working
        return -1, 0.0, ["?", "?", "?"], [0.0, 0.0, 0.0]


def update_display():
    """Update the displayed image and style feedback"""
    generated_image = generate_image()

    # Update main image
    ax_image.clear()
    ax_image.imshow(generated_image.T, cmap="gray", origin="lower")

    letter_char = chr(ord("a") + current_letter)
    style_str = "[" + ", ".join([f"{v:.2f}" for v in current_style_values]) + "]"

    ax_image.set_title(
        f"Generated Letter: '{letter_char}'\n"
        f"Style: {style_str[:50]}{'...' if len(style_str) > 50 else ''}",
        fontsize=14,
        fontweight="bold",
    )
    ax_image.axis("off")

    # Update style predictor feedback
    ax_feedback.clear()
    ax_feedback.axis("off")

    predicted_idx, confidence, top3_chars, top3_confs = get_style_prediction()

    if predicted_idx >= 0:
        predicted_char = chr(ord("a") + predicted_idx)
        true_char = letter_char

        # Determine disentanglement quality
        is_leaking = predicted_idx == current_letter
        random_baseline = 1.0 / 26  # 3.85%

        # Create feedback text
        feedback_text = "STYLE PREDICTOR ANALYSIS\n"
        feedback_text += "=" * 35 + "\n\n"

        feedback_text += f"True Character:     '{true_char}'\n"
        feedback_text += f"Style Predicts:     '{predicted_char}'\n"
        feedback_text += f"Confidence:         {confidence:.1%}\n"
        feedback_text += f"Random Baseline:    {random_baseline:.1%}\n\n"

        feedback_text += "Top 3 Predictions:\n"
        for i, (char, conf) in enumerate(zip(top3_chars, top3_confs)):
            marker = "→" if i == 0 else " "
            feedback_text += f"  {marker} '{char}': {conf:.1%}\n"

        feedback_text += "\n" + "-" * 35 + "\n"

        # Disentanglement status
        if is_leaking and confidence > 0.15:
            feedback_text += "\n⚠ WARNING: Style Leaking!\n\n"
            feedback_text += "Style encodes character info.\n"
            feedback_text += "Try different style values.\n"
            bg_color = "lightcoral"
            quality = "POOR"
        elif confidence > 0.15:
            feedback_text += "\n○ MODERATE Disentanglement\n\n"
            feedback_text += "Style shows some bias but\n"
            feedback_text += "does not match true char.\n"
            bg_color = "lightyellow"
            quality = "MODERATE"
        elif confidence > 0.10:
            feedback_text += "\n✓ GOOD Disentanglement\n\n"
            feedback_text += "Style is mostly independent\n"
            feedback_text += "of character identity.\n"
            bg_color = "lightgreen"
            quality = "GOOD"
        else:
            feedback_text += "\n✓ EXCELLENT Disentanglement!\n\n"
            feedback_text += "Style is highly independent\n"
            feedback_text += "of character identity.\n"
            bg_color = "palegreen"
            quality = "EXCELLENT"

        feedback_text += f"\nQuality: {quality}\n"

        # Display the feedback
        ax_feedback.text(
            0.05,
            0.95,
            feedback_text,
            transform=ax_feedback.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor=bg_color, alpha=0.4, pad=0.8),
        )
    else:
        # Fallback if style predictor not available
        fallback_text = "STYLE PREDICTOR\n"
        fallback_text += "=" * 35 + "\n\n"
        fallback_text += "Style predictor not available.\n\n"
        fallback_text += "Using decoder only.\n"

        ax_feedback.text(
            0.05,
            0.95,
            fallback_text,
            transform=ax_feedback.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
        )

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
    print(current_style_values)
    for i in range(12):
        sliders[i].set_val(current_style_values[i])
    update_display()


# Connect callbacks
radio_all.on_clicked(on_radio_change)
for slider in sliders:
    slider.on_changed(on_slider_change)

# Add control buttons
ax_reset = plt.axes([0.30, 0.02, 0.12, 0.035])
btn_reset = Button(ax_reset, "Reset Style", color="lightcoral", hovercolor="coral")
btn_reset.on_clicked(lambda event: reset_style())

ax_random = plt.axes([0.47, 0.02, 0.12, 0.035])
btn_random = Button(ax_random, "Random Style", color="lightgreen", hovercolor="green")
btn_random.on_clicked(lambda event: randomize_style())

# Initial display
update_display()

print("\n" + "=" * 80)
print("Interactive Adversarial Lowercase Generator Ready!")
print("=" * 80)
print("\nInstructions:")
print("  1. Click radio buttons to select which letter (a-z)")
print("  2. Adjust 12 style sliders (S0-S11) to control appearance")
print("  3. Watch the 'Style Disentanglement Analysis' panel on the right")
print("  4. GREEN = Good disentanglement (style independent of character)")
print("  5. YELLOW/ORANGE = Moderate (some correlation)")
print("  6. RED = Style leaking character info (poor disentanglement)")
print("  7. Use 'Reset Style' for neutral (zero) style")
print("  8. Use 'Random Style' to explore random variations")
print("\nWhat to Look For:")
print("  - Style Predictor should have LOW confidence (< 15%)")
print("  - Style Predictor should NOT match the true character often")
print("  - Random baseline is 3.85% (1/26 classes)")
print("  - If confidence is high, style is encoding character info")
print("\nBenefits of Adversarial Model:")
print("  - Changing style should NOT change the letter")
print("  - Same style applied to different letters gives consistent appearance")
print("  - Style dimensions are more interpretable")
print("  - Better control for creating styled text")
print("\nExperiment:")
print("  - Pick a letter and sweep through style dimensions")
print("  - Check if style predictor confidence stays low")
print("  - Compare same style vector across different letters")
print("  - Look for style dimensions that increase predictor confidence")
print("\nClose the window to exit.")
print("=" * 80)

plt.show()
