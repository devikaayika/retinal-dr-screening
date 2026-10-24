
import tensorflow as tf

print("Loading EfficientNetB0...")

base_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers for the initial training stage
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

# Extract image features
x = base_model(inputs, training=False)

# Convert feature maps into a feature vector
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Regularization
x = tf.keras.layers.Dropout(0.3)(x)

# Five severity classes
outputs = tf.keras.layers.Dense(
    5,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.summary()

print("\nEfficientNetB0 architecture created successfully!")
print("Number of output classes:", model.output_shape[-1])
print("Pretrained base model frozen:", not base_model.trainable)
