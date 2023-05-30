import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten

import json
import numpy as np

# Load the data
with open("data/intents.json") as file:
    data = json.load(file)

# Extract the training data
sentences = []
labels = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# Convert the sentences to sequences
sequences = tokenizer.texts_to_sequences(sentences)

# Pad the sequences
maxlen = 20
padded_sequences = pad_sequences(sequences, maxlen=maxlen)

# Convert the labels to one-hot vectors
labels = np.array(labels)
unique_labels = np.unique(labels)
label_to_index = dict((label, index)
                      for index, label in enumerate(unique_labels))
index_to_label = dict((index, label)
                      for index, label in enumerate(unique_labels))
label_indices = np.array([label_to_index[label] for label in labels])
one_hot_labels = tf.keras.utils.to_categorical(label_indices)

# Define the model
model = Sequential([
    Embedding(len(word_index) + 1, 128, input_length=maxlen),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(unique_labels), activation="softmax")
])

# Compile the model
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(padded_sequences, one_hot_labels,
          epochs=1000, batch_size=8, verbose=1)

# Save the model
model.save("model.h5")
