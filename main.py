from nltk.stem.lancaster import LancasterStemmer
from telegram.constants import ParseMode
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten

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
import json
import requests
import tensorflow as tf
from googleapiclient.discovery import build
from typing import Final
import csv
import nltk
nltk.download("punkt")
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
from word2number import w2n


stemmer = LancasterStemmer()

TOKEN: Final = "6054945876:AAFxijladNrhRgATM4iOwRXcMm_MfuEGBDQ"  # bot token
BOT_USERNAME: Final = '@KBS23bot'
YT_key: Final = 'AIzaSyBxNofXCt8LaoBJri2_lBhJGFWSMXM-oCE'

qStarted: bool = False
qIndex: int = 0  # test question index
similarStarted: bool = False
series = []
finalList = []
# Series age classification
classifications = {
    'A': 0,
    'All': 0,
    '7': 7,
    '7+': 7,
    'PG': 8,
    '12+': 12,
    'UA': 12,
    '13': 13,
    '13+': 13,
    'U': 14,
    '15': 15,
    '15+': 15,
    '16': 16,
    '16+': 16,
    '18': 18,
    '18+': 18,
    'nan': 18,
    'R': 18,
    'Not Rated': 18
}

currentSynonyms = ['current', 'present', 'contemporary', 'late', 'modern', 'new', 'actual', 'recent', 'fresh', 'trendy',
                   'innovative']
classicSynonyms = ['classic', 'classical', 'iconic', 'old', 'vintage', 'legendary', 'past', 'historic', 'historical',
                   'previous', 'early', 'ancient', 'prior', 'old-fashioned']
genres = ['action', 'adventure', 'drama', 'crime', 'thriller', 'horror', 'comedy', 'romance', 'fantasy', 'mystery',
          'sci-fi', 'animation', 'biography', 'family', 'history', 'documentary', 'music', 'sport', 'western', 'war',
          'musical', 'reality-tv', 'short', 'news', 'talk-show', 'game-show']
relevantSynonyms = ['relevant', 'high', 'big', 'important', 'huge', 'elevated', 'giant', 'key', 'dominant', 'supreme',
                    'crucial', 'vital', 'indispensable', 'critical', 'essential', 'significant', 'fundamental',
                    'useful']
irrelevantSynonyms = ['irrelevant', 'small', 'little', 'tiny', 'short', 'minor']

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
          epochs=500, batch_size=8, verbose=1)

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
    await update.message.reply_text(
        "Hi! I'm here to recommend TV Series. Talk to me and discover what I have to offer ;)")


async def helpCommand(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Need help huh?\nYou can just talk to me and i'll try my best to answer something that makes sense.\nIf you ask for a questionaire i'll eventually recommend something\n(I can also give you trailers from tv series, that's cool right?)")


async def stopCommand(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("cya buddy")
    context.bot.stop()


async def getSimilarShows(text: str):
    global similarStarted

    API_KEY = '4ec06d8570075916af7293379cbf5430'

    # Search for the TV show you want to find similar shows to
    url = f"https://api.themoviedb.org/3/search/tv?api_key={API_KEY}&query={text}"
    response = requests.get(url)
    results = response.json()["results"]

    # Get the ID of the first search result
    tv_id = results[0]["id"]

    # Get the similar TV shows
    url = f"https://api.themoviedb.org/3/tv/{tv_id}/similar?api_key={API_KEY}"
    response = requests.get(url)
    similar_shows = response.json()["results"]

    # Print the titles of the three most similar shows
    for show in similar_shows[:3]:
        print(show["name"])
    similarStarted = False

    return similar_shows


# async def resultCommand():

# get all movies from the dataset


def getAllSeries():
    with open('data/series_data.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            serie = {
                'poster': row[0],
                'title': row[1],
                'period': row[2],
                'certificate': row[3],
                'duration': getDuration(row[4]),
                'genre': row[5],
                'imdb_rating': row[6],
                'overview': row[7],
                'star1': row[8],
                'star2': row[9],
                'star3': row[10],
                'star4': row[11],
                'no_of_votes': row[12]
            }
            # Append the serie to the list
            series.append(serie)

    return series


# Responses


async def handleResponse(text: str) -> str:
    processed: str = text.lower()  # convert user input text to lower case
    if 'trailer' in processed:
        return search_video(processed, YT_key)
    elif processed == "/exit":
        return "Exiting questionnaire..."
    else:
        message = predict_tag(processed)
        return message


async def handleMessage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text: str = update.message.text  # user input
    global qStarted
    global qIndex
    global similarStarted
    questions = ["How old are you?", "How long are the episodes of the series you usually watch? (in minutes)",
                 "Do you prefer current or classic series?", "What genres are you looking for? \U0001F3AD",
                 "How relevant is the opinion of critics to you? \U0001F4DD\U0001F440"]
    if text == "/questions":
        qStarted = True
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text='I will ask you a few questions to know more about you \U0001F50E\nStarting questionnaire...')
        await update.message.reply_text(questions[qIndex])
    elif text == "/similar":
        similarStarted = True
        await update.message.reply_text("Give me the name of a tv-show please: ")
    elif similarStarted:
        print("similar ongoing")
        similarShows = await getSimilarShows(text)
        for show in similarShows[:3]:
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text="*" + show["name"] + "* \u2935\uFE0F",
                                           parse_mode=ParseMode.MARKDOWN)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="https://image.tmdb.org/t/p/w500" +
                                                                                  show["poster_path"])
            await context.bot.send_message(chat_id=update.effective_chat.id, text=show["overview"])
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text=search_video("Trailer of " + show["name"], YT_key))
    elif qStarted:
        answer: str = await handleQuestion(text)
        result = processResponse(answer, qIndex)
        if result is not None:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=result)
        else:
            qIndex = qIndex + 1
        if qIndex < 5:
            await update.message.reply_text(questions[qIndex])
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text="That's the end of the questionnaire.\nYou got a \U0001F4AF\nHere are my recommendations \U0001F60F")

            finalSortedList = sorted(finalList, key=lambda x: x['imdb_rating'], reverse=True)
            for i in range(0, 3, 1):
                await context.bot.send_message(chat_id=update.effective_chat.id,
                                               text="*" + finalSortedList[i]['title'] + "* \u2935\uFE0F",
                                               parse_mode=ParseMode.MARKDOWN)
                await context.bot.send_message(chat_id=update.effective_chat.id, text=finalSortedList[i]['poster'])
                await context.bot.send_message(chat_id=update.effective_chat.id, text=finalSortedList[i]['overview'])
                await context.bot.send_message(chat_id=update.effective_chat.id,
                                               text=search_video("Trailer of " + finalSortedList[i]['title'], YT_key))
            qStarted = False
            qIndex = 0

    else:
        response: str = await handleResponse(text)
        print('Bot:', response)
        await update.message.reply_text(response)


async def handleQuestion(input: str):
    # detect what has been asked and prune tv series
    # reply with text
    # use global variables for this, to detect if questions finished and detect question index in array
    global qIndex

    return input


# Process questionnaire responses
def getNum(word):
    try:
        num = w2n.word_to_num(word)
        return num
    except ValueError:
        return None


def isYear(year):
    try:
        datetime.strptime(str(year), "%Y")
        return True
    except ValueError:
        return False


def getCertificate(response):
    tokens = nltk.word_tokenize(response)
    age = None
    for token in tokens:
        age = getNum(token)
        if age is not None:
            break

    if isYear(age):
        age = datetime.now().year - age

    if age is not None:
        eligibleClass = []
        for classification, appropAge in classifications.items():
            if age >= appropAge:
                eligibleClass.append(classification)
        return eligibleClass
    else:
        return age


def getDuration(response):
    tokens = nltk.word_tokenize(response)
    for token in tokens:
        time = getNum(token)
        if time is not None:
            return time

    return None


def determinePreference(response):
    tokens = word_tokenize(response.lower())
    stopWords = set(stopwords.words('english'))
    filteredTokens = [token for token in tokens if token not in stopWords]
    lemmatizer = WordNetLemmatizer()
    lemmatizedTokens = [lemmatizer.lemmatize(token, pos='a') for token in filteredTokens]

    for token in lemmatizedTokens:
        if token in currentSynonyms:
            return "current"
        elif token in classicSynonyms:
            return "classic"

    return None


def isPositive(response):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(response)
    if sentiment['compound'] >= 0:
        return True
    else:
        return False


def checkFinalListDecade(preference):
    for serie in reversed(finalList):
        if (preference == "current" and 2000 > int(serie['period'].split('–')[0].strip('()'))) or \
                (preference == "classic" and 2000 <= int(serie['period'].split('–')[0].strip('()'))):
            finalList.remove(serie)


def getGenres(response):
    genreList = []
    tokens = word_tokenize(response.lower())
    stopWords = set(stopwords.words('english'))
    filteredTokens = [token for token in tokens if token not in stopWords]

    if isPositive(response):
        for token in filteredTokens:
            if token in genres:
                genreList.append(token)
    else:
        genreList = genres
        for token in filteredTokens:
            if token in genreList:
                genreList.remove(token)

    return genreList if genreList else None


def filterByGenre(genreList):
    match = False
    for serie in reversed(finalList):
        serieGenres = serie['genre'].split(", ")
        serieGenres = [genre.lower() for genre in serieGenres]

        for genre in serieGenres:
            if genre in genreList:
                match = True

        if not match:
            finalList.remove(serie)
        else:
            match = False


def isRelevant(response):
    if isPositive(response):
        tokens = word_tokenize(response.lower())
        stopWords = set(stopwords.words('english'))
        filteredTokens = [token for token in tokens if token not in stopWords]
        lemmatizer = WordNetLemmatizer()
        lemmatizedTokens = [lemmatizer.lemmatize(token, pos='a') for token in filteredTokens]

        for token in lemmatizedTokens:
            if token in relevantSynonyms:
                return True
            elif token in irrelevantSynonyms:
                return False

        return None
    else:
        return False


def filterByRating():
    for serie in finalList:
        if float(serie['imdb_rating']) < 7.5:
            finalList.remove(serie)


def processResponse(response: str, question: int):
    if question == 0:
        userCertificate = getCertificate(response)
        if userCertificate is None:
            return "From the answer given I cannot determine your age"
        else:
            # Get series having that certificate
            finalList.clear()
            for serie in series:
                if serie['certificate'] in userCertificate:
                    finalList.append(serie)
            return None
    elif question == 1:
        userDuration = getDuration(response)
        if userDuration is None:
            return "Tell me a specific duration value"
        else:
            for serie in reversed(finalList):
                if serie['duration'] is None or abs(userDuration - serie['duration']) > 10:
                    finalList.remove(serie)
            return None
    elif question == 2:
        userPreference = determinePreference(response)
        if userPreference is None:
            return "This is not what I am asking \U0001F914"
        else:
            if not isPositive(response):
                userPreference = "classic" if userPreference == "current" else "current"
            checkFinalListDecade(userPreference)
            return None
    elif question == 3:
        userGenres = getGenres(response)
        if userGenres is None:
            return "I don't have this genre on my list \U0001F615"
        else:
            filterByGenre(userGenres)
            return None
    elif question == 4:
        criticsRelevance = isRelevant(response)
        if criticsRelevance is None:
            return "I don't get what you are trying to say \U0001F62C\U0001F937"
        else:
            if criticsRelevance is True:
                filterByRating()
            return None


# when the user inputs the command /questions, start the questionaire


async def startQuestionnaire(update: Update, context: ContextTypes.DEFAULT_TYPE):
    questions = ["q1", "q2", "q3"]
    answers = []
    global qStarted
    qStarted = True
    while qStarted:
        for q in questions:
            await update.message.reply_text(q)
            # answer = await context.bot.await_message(chat_id=update.message.chat_id, timeout=30)
            answer = update.message.text
            print(answer)
            if answer is None:
                await update.message.reply_text("Time's up! Exiting questionnaire...")
                qStarted = False
                return
            elif answer.text.lower() == "exit":
                await update.message.reply_text("Exiting questionnaire...")
                qStarted = False
                return
            answers.append(answer.text)
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
    getAllSeries()
    print('Starting bot...')
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('start', startCommand))
    app.add_handler(CommandHandler('help', helpCommand))
    app.add_handler(CommandHandler('stop', stopCommand))

    # Messages
    app.add_handler(MessageHandler(filters.Text(), handleMessage))

    app.add_handler(MessageHandler(filters.Text(), handleQuestion))

    # Errors
    app.add_error_handler(error)

    # Polling bot
    print('Started polling...')
    app.run_polling(poll_interval=3)
