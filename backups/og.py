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
import tflearn
from googleapiclient.discovery import build
from typing import Final
import csv
import nltk
stemmer = LancasterStemmer()

TOKEN: Final = "6072166464:AAGK_3kc39vEAGwhNR2on90NZ95KXNNpQ-Y"  # bot token
BOT_USERNAME: Final = '@MecagoensatanasBot'
YT_key: Final = 'AIzaSyBxNofXCt8LaoBJri2_lBhJGFWSMXM-oCE'


# Commands


async def startCommand(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("hello, i recommend tv series")


async def helpCommand(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("answer the questions so i can recommend tv series ")


async def stopCommand(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("cya buddy")
    context.bot.stop()


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
    processed: str = text.lower()  # convert text to lower case
    analyzeWithSpacy(processed)
    if 'hello' in processed:
        return 'hi buddy'
    if 'trailer' in processed:
        return search_video(processed, YT_key)
    if 'how many' in processed:
        return 'i have a database with +2000 tv series'
    else:
        return 'i dont understand'


async def handleMessage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text: str = update.message.text
    response: str = await handleResponse(text)
    print('Bot:', response)
    await update.message.reply_text(response)

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

# spaCy


def analyzeWithSpacy(user_input: str):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(user_input)
    for entity in doc.ents:
        print(entity.label)
        if entity.label_ == "FILM":
            print("im talking about movies")
            return True
            print("Movie mentioned:", entity.text)
            # Perform further analysis based on the movie entity
            # For example, you can check if the user expressed a preference or dislike for the movie
            if "love" in user_input.lower():
                print("User expressed love for the movie.")
            elif "dislike" in user_input.lower():
                print("User expressed dislike for the movie.")


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

    #print(search_video('game of thrones trailer', YT_key))

    # Polling bot
    print('Started polling...')
    app.run_polling(poll_interval=3)
