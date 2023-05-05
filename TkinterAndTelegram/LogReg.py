import pickle
import json
import random
import nltk
from nltk.corpus import stopwords
from textblob import Word

nltk.download('stopwords')
nltk.download('wordnet')
stop = stopwords.words('english')
emotionsSVMLOGREG = ["anger", "happy", "love", "neutral", "sad"]
emotion_gauge = {"anger": 0, "happy": 0, "love": 0, "neutral": 0, "sad": 0}
responsesJSON = json.load(open('responses.json', encoding="utf8"))

with open('LogRegCountVectorizer1.0', 'rb') as LogRegFiles:
    logreg, CVectorizer = pickle.load(LogRegFiles)
    vectorizer = CVectorizer
    model = logreg

def preprocessMessageSVMLOGREG(message):
    message = ' '.join([word for word in message.split() if word not in stop])
    message = " ".join([Word(word).lemmatize() for word in message.split()])
    messageCountVector = vectorizer.transform([message])
    return messageCountVector

def getPredictionScoresSVMLOGREG(messageCountVector):
    prediction = model.predict_proba(messageCountVector)
    predictionScores = prediction[0]
    return predictionScores

def updateUserEmotionsSVMLOGREG(predictionScores):
    # In SVM and LOGREG - anger, happy, love, neutral, sad
    emotion_gauge['anger'] += predictionScores[0]
    emotion_gauge['happy'] += predictionScores[1]
    emotion_gauge['love'] += predictionScores[2]
    emotion_gauge['neutral'] += predictionScores[3]
    emotion_gauge['sad'] += predictionScores[4]
    currentMemory = emotion_gauge
    return currentMemory

def getEmotionFeltSVMLOGREG(predictionScores):
    highest_score = max(predictionScores)
    prediction_list = list(predictionScores)
    indexOfMaxEmotion = prediction_list.index(highest_score)
    emotion_felt = emotionsSVMLOGREG[indexOfMaxEmotion]
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
    elif message.lower() == "/logreg":
        outputFromModel = "Logistic Regression Model Loaded for Emotion Detection!"
    else:
        messageCountVector = preprocessMessageSVMLOGREG(message)
        predictionScores = getPredictionScoresSVMLOGREG(messageCountVector)
        currentMemory = updateUserEmotionsSVMLOGREG(predictionScores)
        print("Updated Emotion Dictionary / MEMORY : ", currentMemory)
        print("\n")
        getEmotionDetected = getEmotionFeltSVMLOGREG(predictionScores)
        print("Emotion Identified :", getEmotionDetected)
        print("------------------------------------------------------")
        outputFromModel = getReply(getEmotionDetected)
    return outputFromModel