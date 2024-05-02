import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input

model = tf.keras.models.load_model('VGG16_model.h5')


def load_and_preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)


def predict(image_path):
    processed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(processed_image)

    predicted_class = 'Biryani' if prediction[0][0] < 0.5 else 'Noodles'

    return predicted_class


if __name__ == '__main__':
    path = 'test_images/noodles.jpg'
    prediction = predict(path)
    print(f"The image is classified as: {prediction}")
