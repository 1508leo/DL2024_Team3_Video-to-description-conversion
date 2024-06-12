import numpy as np
import cv2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image

from google.colab import files
uploaded = files.upload()

# InceptionV3
model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

def extract_features(video_path):
    # Add video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # 
    sample_frames = frames[::int(len(frames)/5)]

    # Feature extraction
    features = []
    for frame in sample_frames:
        img = cv2.resize(frame, (299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        features.append(feature.flatten())
    return np.array(features)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, TimeDistributed, RepeatVector, Activation
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Video and description
video_features = extract_features('test_videodatainfo.json') 
descriptions = 'VideoDescriptions.txt'    # 

# Vocabulary
tokenizer = Tokenizer()
tokenizer.fit_on_texts(descriptions)
vocab_size = len(tokenizer.word_index) + 1

# Sentence
sequences = tokenizer.texts_to_sequences(descriptions)
max_sequence_length = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Sequential model
model = Sequential()
model.add(LSTM(256, input_shape=(5, 2048), return_sequences=True))
model.add(LSTM(256))
model.add(Dense(256, activation='relu'))
model.add(RepeatVector(max_sequence_length))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# 
X = np.array(video_features)
y = np.expand_dims(sequences, axis=-1)

model.fit(X, y, epochs=50, batch_size=64)

# Generate description
def generate_description(video_features):
    prediction = model.predict(np.array([video_features]))
    sequence = np.argmax(prediction, axis=-1)
    words = [tokenizer.index_word[idx] for idx in sequence[0] if idx > 0]
    return ' '.join(words)

# Training
test_video_path = 'path_to_test_video.mp4'
test_features = extract_features(test_video_path)
description = generate_description(test_features)
print(description)