from flask import *
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

encoder = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

def encode(text):
    """Embeds a text string using the Universal Sentence Encoder."""
    if isinstance(text, str) and text.strip() != '':
        return encoder([text]).numpy().tolist()  # Convert numpy array to list
    else:
        return np.zeros((1, 512)).tolist()  # Return a zero vector as a list if the text is empty.

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the Text Similarity API <br> Please provide a json file using a POST request"

@app.route('/', methods=['POST'])
def similar():
    data = request.get_json()
    text1 = data["text1"]
    text2 = data["text2"]
    encoding1 = np.array(encode(text1))  # Convert list to numpy array
    encoding2 = np.array(encode(text2))  # Convert list to numpy array
    similarity = np.dot(encoding1, encoding2.transpose()) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
    return json.dumps({'similarity score': round(similarity.item(), 1)})  # Convert similarity to a scalar value

if __name__ == '__main__':
    app.run(debug=True)
