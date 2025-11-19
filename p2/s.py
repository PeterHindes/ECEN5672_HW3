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