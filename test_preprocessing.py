import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Create a sample image tensor to test resizing
sample_image = tf.zeros((2136, 3216, 3), dtype=tf.float32)

resized_image = tf.image.resize(sample_image, (224, 224))

print("Original image shape:", sample_image.shape)
print("Resized image shape:", resized_image.shape)
print("Preprocessing test completed!")
