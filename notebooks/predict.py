from flask import Flask, request, jsonify
import requests
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime;
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
   
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64") 
        label_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        # At test time, just return the computed predictions.
        return y_pred


# Register the custom layer
tf.keras.utils.get_custom_objects()['CTCLayer'] = CTCLayer
# Load the model
model = keras.models.load_model("handwriting.h5")
characters = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
AUTOTUNE = tf.data.AUTOTUNE
# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]
    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image
batch_size = 64
padding_token = 99
image_width = 128
image_height = 32

def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def calculate_edit_distance(labels, predictions):
    # Get a single batch and convert its labels to sparse tensors.
    saprse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)
    # Make predictions and convert them to sparse tensors.
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = keras.backend.ctc_decode(
        predictions, input_length=input_len, greedy=True
    )[0][0][:, :max_len]
    sparse_predictions = tf.cast(
        tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
    )
    # Compute individual edit distances and average them out.
    edit_distances = tf.edit_distance(
        sparse_predictions, saprse_labels, normalize=False
    )
    return tf.reduce_mean(edit_distances)
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
max_len = 71
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def prepare_dataset_custom(img_paths):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths)).map(
        preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

# def download_image(url, destination_path):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Check if the request was successful

#         with open(destination_path, 'wb') as file:
#             file.write(response.content)

#         print(f"Image downloaded successfully to: {destination_path}")

#     except requests.exceptions.RequestException as e:
#         print(f"Error downloading image: {e}")

from PIL import Image, ImageEnhance
from io import BytesIO

def download_image(url, destination_path):
    try:
        # Download the image
        response = requests.get(url)
        response.raise_for_status()

        # Open the image using Pillow
        with Image.open(BytesIO(response.content)) as img:
            # Enhance the brightness of the image
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.5)  # Adjust the factor as needed

            # Convert the image to grayscale
            img = img.convert('L')

            # Save the preprocessed image
            img.save(destination_path)

        print(f"Image preprocessed and saved successfully to: {destination_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")

@app.route('/get_text', methods=['POST'])
def get_text():
    try:
        data = request.json
        # image_url = 'https://firebasestorage.googleapis.com/v0/b/smart-track-32319.appspot.com/o/prediction_images%2F1407.jpg?alt=media&token=3182fa63-3876-435e-92f3-918c5cf0082e'
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        image_path = f"{timestamp}_image.png"
        download_image(data['image'],image_path)

        # Call your existing functions
        img_paths = [image_path]
        custom_ds = prepare_dataset_custom(img_paths)

        for batch in custom_ds.take(1):
            preds = prediction_model.predict(batch)
            pred_texts = decode_batch_predictions(preds)
            result = pred_texts[0]
            # Return the result as JSON
            return jsonify({'prediction': result})

    except Exception as e:
        # Handle any errors and return as JSON
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)