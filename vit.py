import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Set Parameters
IMG_SIZE = 69  # Image size (69x69)
PATCH_SIZE = 9  # Patch size (each patch is 9x9)
NUM_CLASSES = 3  # Galaxy classes: E, S, SB
D_MODEL = 64  # Embedding size
NUM_HEADS = 4  # Number of attention heads
NUM_LAYERS = 4  # Transformer Encoder layers

# Load Dataset (Replace paths)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "path/to/train_dataset",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    label_mode="int"
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "path/to/val_dataset",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    label_mode="int"
)

# Normalize images
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    return image, label

train_ds = train_ds.map(preprocess).shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).prefetch(buffer_size=tf.data.AUTOTUNE)

# Create Patch Embedding Layer
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = layers.Dense(embed_dim)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        num_patches = patches.shape[1] * patches.shape[2]
        patches = tf.reshape(patches, (batch_size, num_patches, -1))
        return self.proj(patches)

# Create Transformer Encoder
def transformer_encoder(embed_dim, num_heads):
    inputs = layers.Input(shape=(None, embed_dim))
    x = layers.LayerNormalization()(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = layers.Add()([inputs, x])
    x = layers.LayerNormalization()(x)
    x = layers.Dense(embed_dim * 2, activation="relu")(x)
    x = layers.Dense(embed_dim)(x)
    outputs = layers.Add()([x, inputs])
    return keras.Model(inputs, outputs)

# Build ViT Model
def build_vit(img_size=69, patch_size=9, embed_dim=64, num_heads=4, num_layers=4, num_classes=3):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = PatchEmbedding(patch_size, embed_dim)(inputs)

    # Positional Encoding
    num_patches = (img_size // patch_size) ** 2
    pos_encoding = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)(tf.range(num_patches))
    x += pos_encoding

    # Transformer Encoder Stack
    for _ in range(num_layers):
        x = transformer_encoder(embed_dim, num_heads)(x)

    # Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model

# Compile and Train ViT Model
vit_model = build_vit()
vit_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = vit_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# Save Model
vit_model.save("galaxy_vit_model.h5")