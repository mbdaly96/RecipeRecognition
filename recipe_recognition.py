import streamlit as st
# import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
from keras.applications.inception_v3 import preprocess_input, InceptionV3
from keras.utils import pad_sequences
import pickle
from keras.models import Model
# from PIL import Image

with open('word_to_index.pickle', 'rb') as handle:
    word_to_index = pickle.load(handle)

index_to_word = dict(map(reversed, word_to_index.items()))
# Load Models
img_embedder = InceptionV3()
img_embedder = Model(img_embedder.input, img_embedder.layers[-2].output)
# img_embedder = load_model('inception_image_embedder.h5', compile=False)
model = load_model('recipeTitler_e100_s800_b128_80p_lr0001.h5', compile=False)

st.title('Recipe Recgonition')

img = st.file_uploader("Upload an image...")

if img is not None:
    # Preprocess Image
    st.image(img, caption='Input Image', use_column_width=True)
    img = np.asarray(bytearray(img.read()), dtype=np.uint8)
    img = cv2.imdecode(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    image_input = np.reshape(img_embedder(img), (1,2048))
    title = 'startSeq'
    for i in range(1, 20):
        seq = [word_to_index[word] for word in title.split() if word in word_to_index]
        seq = pad_sequences([seq], maxlen=20)
        pred = model.predict(x=[image_input, seq])
        pred = np.argmax(pred[0])
        title += ' ' + index_to_word[pred]
        if index_to_word[pred] == 'endSeq':
            break

    title = title.split()[1:-1]
    title = ' '.join(title)
    st.write(title)
