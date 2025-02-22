import os

from keras.src.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model

img_height = 69
img_width = 69
batch_size = 32
train_dir = "images_E_S_SB_69x69_a_03/images_E_S_SB_69x69_a_03_train"
train_ds = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names


def predict_galaxy(model, img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array = img_array / 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    class_name = train_ds.class_names[predicted_class[0]]
    return class_name


model_path = "C:\\Users\\nerdi\\Desktop\\CNNGalaxy\\galaxy_classifier.h5"
print("Model exists:", os.path.exists(model_path))
model = load_model("C:\\Users\\nerdi\\Desktop\\CNNGalaxy\\galaxy_classifier.h5")
# Example usage
print(predict_galaxy(model, '6790.jpg'))
