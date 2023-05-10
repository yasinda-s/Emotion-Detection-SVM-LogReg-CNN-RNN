from telegram.ext import *
import pickle
import json
import random
import nltk
from nltk.corpus import stopwords
from textblob import Word
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

API_KEY = '6031338657:AAFsHFZLkI-crEluI4Ryhh_BB-c2Jayq7Us'
updater = Updater(API_KEY)
nltk.download('stopwords')
nltk.download('wordnet')
stop = stopwords.words('english')
MAX_SEQUENCE_LENGTH = 30
emotionsHNN = ["neutral", "happy", "sad", "love", "anger"]
responsesJSON = json.load(open('responses.json', encoding="utf8"))
emotion_gauge = {"anger": 0, "happy": 0, "love": 0, "neutral": 0, "sad": 0}

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model("HNNWeights.h5")


def preprocessMessageHNN(message):
    message = ' '.join([word for word in message.split() if word not in stop])
    message = " ".join([Word(word).lemmatize() for word in message.split()])
    message = [message]
    sequences = tokenizer.texts_to_sequences(message)
    pre_padding = pad_sequences(sequences, padding='pre', maxlen=(MAX_SEQUENCE_LENGTH - 5))
    post_padding = pad_sequences(pre_padding, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
    return post_padding


def getPredictionScoresHNN(messagePostPadding):
    prediction = model.predict(messagePostPadding)
    predictionScores = prediction[0]
    return predictionScores


def updateUserEmotionsHNN(predictionScores):
    # In HNN - neutral, happy, sad, love, anger
    emotion_gauge['anger'] += predictionScores[4]
    emotion_gauge['happy'] += predictionScores[1]
    emotion_gauge['love'] += predictionScores[3]
    emotion_gauge['neutral'] += predictionScores[0]
    emotion_gauge['sad'] += predictionScores[2]
    currentMemory = emotion_gauge
    return currentMemory


def getEmotionFeltHNN(predictionScores):
    highest_score = max(predictionScores)
    prediction_list = list(predictionScores)
    indexOfMaxEmotion = prediction_list.index(highest_score)
    emotion_felt = emotionsHNN[indexOfMaxEmotion]
    return emotion_felt


def getReply(emotion_felt):
    if emotion_felt == "anger":
        replyPrefix = "Emotion Detected - Anger (Negative)\n"
    elif emotion_felt == "happy":
        replyPrefix = "Emotion Detected - Happy (Positive)\n"
    elif emotion_felt == "love":
        replyPrefix = "Emotion Detected - Love (Positive)\n"
    elif emotion_felt == "neutral":
        replyPrefix = "Emotion Detected - Neutral\n"
    elif emotion_felt == "sad":
        replyPrefix = "Emotion Detected - Sad (Negative)\n"
    for i in range(5):
        if responsesJSON['responses_outer'][i]['emotion'] == emotion_felt:
            replySuffix = str(responsesJSON['responses_outer'][i]['responses'][
                                  random.randrange(0, len(responsesJSON['responses_outer'][i]['responses']))])
    reply = replySuffix
    return reply


def finalMessage():
    highestEmotion = max(emotion_gauge, key=emotion_gauge.get)
    if highestEmotion == "anger":
        finalReply = "The overall emotions you are feeling seem to be negative. It sounds like you are quite frustrated with your life right now, I recommend explaining your thoughts to a friend. I'm here if you want to talk further."
    elif highestEmotion == "happy":
        finalReply = "The overall emotions you feel are positive!You seem to be very happy with your life! I love that for you and I hope you continue to feel this way!"
    elif highestEmotion == "love":
        finalReply = "The overall emotions you feel are positive! Its clear that you feel a lot of love and I wish you the best in life! I hope you stay this way!"
    elif highestEmotion == "neutral":
        finalReply = "You seem to be a person that has their feelings in check, I wish you the best in life!"
    elif highestEmotion == "sad":
        finalReply = "The overall emotions you are feeling seem to be negative. It's okay to be a little sad, just know that a lot of people love and care for you! Reach out to them, you are not alone."
    return finalReply


def get_response(message):
    if message.lower() == "/quit":
        outputFromModel = finalMessage()
        # emotion_gauge = {"anger": 0, "happy": 0, "love": 0, "neutral": 0, "sad": 0}
    else:
        messagePostPadding = preprocessMessageHNN(message)
        predictionScores = getPredictionScoresHNN(messagePostPadding)
        currentMemory = updateUserEmotionsHNN(predictionScores)
        print("Updated Emotion Dictionary / MEMORY : ", currentMemory)
        print("\n")
        getEmotionDetected = getEmotionFeltHNN(predictionScores)
        print("Emotion Identified :", getEmotionDetected)
        print("------------------------------------------------------")
        outputFromModel = getReply(getEmotionDetected)
    return outputFromModel


def startBot(update, param):
    name = update.message.chat.first_name
    update.message.reply_text('Hi ' + name + '!')
    update.message.reply_text('I am CompaNeo')
    update.message.reply_text('How are you doing today?')


def botHelp(update, param):
    update.message.reply_text('You can use /start to begin')
    update.message.reply_text("You can type 'quit' to end")


def getUserMessage(update, param):
    message = str(update.message.text).lower()
    response = get_response(message)
    update.message.reply_text(response)


def errorMessage(update, context):
    print(f"Update{update} caused error {context.error}")


def RunTelegramBot():
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", startBot))
    dp.add_handler(CommandHandler("help", botHelp))
    dp.add_handler(MessageHandler(Filters.text, getUserMessage))
    dp.add_error_handler(errorMessage)
    updater.start_polling()
    updater.idle()


RunTelegramBot()
