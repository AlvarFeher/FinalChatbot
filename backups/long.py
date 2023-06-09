from nltk.stem.lancaster import LancasterStemmer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten

import json
import numpy as np
from telegram import *
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
import spacy
import json
import random
import tensorflow as tf
from googleapiclient.discovery import build
from typing import Final
import csv
import nltk
stemmer = LancasterStemmer()

TOKEN: Final = "6072166464:AAGK_3kc39vEAGwhNR2on90NZ95KXNNpQ-Y"  # bot token
BOT_USERNAME: Final = '@MecagoensatanasBot'
YT_key: Final = 'AIzaSyBxNofXCt8LaoBJri2_lBhJGFWSMXM-oCE'

qStarted: bool = False

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

# predict to which category the input belongs to


def predict_tag(sentence):
    # Convert the sentence to a sequence
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    # Predict the tag of the sentence
    prediction = model.predict(padded_sequence)[0]
    predicted_label_index = np.argmax(prediction)
    predicted_label = index_to_label[predicted_label_index]
    # Return a sentence based on the predicted tag
    for intent in data["intents"]:
        if intent["tag"] == predicted_label:
            return np.random.choice(intent["responses"])

# Commands


async def startCommand(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I'm here to recommend TV Series. Talk to me and discover what I have to offer ;)")


async def helpCommand(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Need help huh?\nYou can just talk to me and i'll try my best to answer something that makes sense.\nIf you ask for a questionaire i'll eventually recommend something\n(I can also give you trailers from tv series, that's cool right?)")


async def stopCommand(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("cya buddy")
    context.bot.stop()

# async def resultCommand():

# get all movies from the dataset


def getAllMovies():
    movies = []
    with open('data/series_data.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            movie = {
                'episodes': row[0],
                'genre': row[1],
                'imdb_rating': row[2],
                'overview': row[3],
                'star1': row[4],
                'star2': row[5],
                'star3': row[6],
                'star4': row[7],
                'no_of_votes': row[8]
            }
            movies.append(movie)
            # print(movie)
        # Append the movie to the list

    return movies

# Responses


async def handleResponse(text: str) -> str:
    processed: str = text.lower()  # convert user input text to lower case
    message = predict_tag(processed)
    if 'trailer' in processed:
        return search_video(processed, YT_key)
    if message == "Okay, I will start":
        print("start asking defined questions")
        qStarted = True
        startQuestionnaire()
        return message
    return message


async def handleMessage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text: str = update.message.text
    if text == "/questions" or qStarted:
        await update.message.reply_text("Starting questionnaire...")
        await startQuestionnaire(update, context)
    else:
        response: str = await handleResponse(text)
        print('Bot:', response)
        await update.message.reply_text(response)


# when the user inputs the command /questions, start the questionaire
async def startQuestionnaire(update: Update, context: ContextTypes.DEFAULT_TYPE):
    questions = ["q1", "q2", "q3"]
    answers = []
    qStarted = True
    for q in questions:
        await update.message.reply_text(q)
        # answer = (await context.bot.await_messages(update.message.chat_id, timeout=30)).text
        answer = (await context.bot.await_messages(update.message.chat_id, timeout=30)).text
        # await answer
        if answer.lower() == "/exit":
            await update.message.reply_text("Exiting questionnaire...")
            qStarted = False
            return
        answers.append(answer)
    await update.message.reply_text("Thank you for answering the questions!")
    print(answers)


# YouTube API
def search_video(title, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Call the search.list method to search for videos with the given title
    search_response = youtube.search().list(
        q=title,
        part='id',
        maxResults=1  # Adjust this number if you want more results
    ).execute()

    # Extract the video ID from the search response
    video_id = search_response['items'][0]['id']['videoId']

    # Create the YouTube video link
    video_link = f'https://www.youtube.com/watch?v={video_id}'

    return video_link

# Errors


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'error caused by {update}: {context.error}')

# Main
if __name__ == '__main__':
    movies = getAllMovies()
    print('Starting bot...')
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('start', startCommand))
    app.add_handler(CommandHandler('help', helpCommand))
    app.add_handler(CommandHandler('stop', stopCommand))

    # Messages
    app.add_handler(MessageHandler(filters.Text(), handleMessage))

    # Errors
    app.add_error_handler(error)

    # Polling bot
    print('Started polling...')
    app.run_polling(poll_interval=3)
