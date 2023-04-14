import pickle
import nltk
from nltk.corpus import stopwords
from textblob import Word

nltk.download('stopwords')
nltk.download('wordnet')

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Contains all the logic of the chatbot
class Chatbot:
    def __init__(self, option):
        print("Option chosen was " + option)
        print("------------------------------------------------------")
        self.option = option
        self.stop = stopwords.words('english')
        self.vectorizer = "Not Set"
        self.text = "Not Set"
        self.model = "Not Set"
        self.tokenizer = "Not Set"
        self.MAX_SEQUENCE_LENGTH = 30
        self.emotionsSVMLOGREG = ["anger", "happy", "love", "neutral", "sad"]
        self.emotionsCNN = ["neutral", "happy", "sad", "love", "anger"]
        self.emotion_gauge = {"anger": 0, "happy": 0, "love": 0, "neutral": 0, "sad": 0}

        if option == "SVM":
            with open('SVMCountVectorizer', 'rb') as SVMFiles:
                svm_model, CVectorizer = pickle.load(SVMFiles)
                self.vectorizer = CVectorizer
                self.model = svm_model
        elif option == "LOGREG":
            with open('LogRegCountVectorizer', 'rb') as LogRegFiles:
                logreg, CVectorizer = pickle.load(LogRegFiles)
                self.vectorizer = CVectorizer
                self.model = logreg
        elif option == "CNN":
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            self.model = load_model("BESTCNNWeightsNLTP.h5")

    def preprocessMessageSVMLOGREG(self, message):
        message = message.replace('[^\w\s]', ' ')
        message = ' '.join([word for word in message.split() if word not in self.stop])
        message = " ".join([Word(word).lemmatize() for word in message.split()])
        messageCountVector = self.vectorizer.transform([message])
        return messageCountVector

    def preprocessMessageCNN(self, message):
        message = message.replace('[^\w\s]', ' ')
        message = ' '.join([word for word in message.split() if word not in self.stop])
        message = " ".join([Word(word).lemmatize() for word in message.split()])
        message = [message]
        sequences = self.tokenizer.texts_to_sequences(message)
        pre_padding = pad_sequences(sequences, padding='pre', maxlen=(self.MAX_SEQUENCE_LENGTH - 5))
        post_padding = pad_sequences(pre_padding, padding='post', maxlen=self.MAX_SEQUENCE_LENGTH)
        return post_padding

    def getPredictionScoresSVMLOGREG(self, messageCountVector):
        prediction = self.model.predict_proba(messageCountVector)
        predictionScores = prediction[0]
        return predictionScores

    def getPredictionScoresCNN(self, messagePostPadding):
        prediction = self.model.predict(messagePostPadding)
        predictionScores = prediction[0]
        return predictionScores

    def updateUserEmotionsSVMLOGREG(self, predictionScores):
        # In SVM and LOGREG - anger, happy, love, neutral, sad
        self.emotion_gauge['anger'] += predictionScores[0]
        self.emotion_gauge['happy'] += predictionScores[1]
        self.emotion_gauge['love'] += predictionScores[2]
        self.emotion_gauge['neutral'] += predictionScores[3]
        self.emotion_gauge['sad'] += predictionScores[4]
        currentMemory = self.emotion_gauge
        return currentMemory

    def updateUserEmotionsCNN(self, predictionScores):
        # In CNN - neutral, happy, sad, love, anger
        self.emotion_gauge['anger'] += predictionScores[4]
        self.emotion_gauge['happy'] += predictionScores[1]
        self.emotion_gauge['love'] += predictionScores[3]
        self.emotion_gauge['neutral'] += predictionScores[0]
        self.emotion_gauge['sad'] += predictionScores[2]
        currentMemory = self.emotion_gauge
        return currentMemory

    def getEmotionFeltSVMLOGREG(self, predictionScores):
        highest_score = max(predictionScores)
        prediction_list = list(predictionScores)
        indexOfMaxEmotion = prediction_list.index(highest_score)
        emotion_felt = self.emotionsSVMLOGREG[indexOfMaxEmotion]
        return emotion_felt

    def getEmotionFeltCNN(self, predictionScores):
        highest_score = max(predictionScores)
        prediction_list = list(predictionScores)
        indexOfMaxEmotion = prediction_list.index(highest_score)
        emotion_felt = self.emotionsCNN[indexOfMaxEmotion]
        return emotion_felt

    @staticmethod
    def getReply(emotion_felt):
        if emotion_felt == "anger":
            reply = "I detect that you are Angry"
        elif emotion_felt == "happy":
            reply = "I detect that you are Happy"
        elif emotion_felt == "love":
            reply = "I detect that you are in Love."
        elif emotion_felt == "neutral":
            reply = "I detect that you are feeling normal."
        elif emotion_felt == "sad":
            reply = "I detect that you are Sad"
        return reply

    @staticmethod
    def finalMessage(emotion_gauge):
        highestEmotion = max(emotion_gauge, key=emotion_gauge.get)
        if highestEmotion == "anger":
            finalReply = "Hey it sounds like you are quite frustrated with your life right now, I recommend explaining your thoughts to a friend. I'm here if you want to talk further."
        elif highestEmotion == "happy":
            finalReply = "You seem to be very happy with your life! I love that for you and I hope you continue to feel this way!"
        elif highestEmotion == "love":
            finalReply = "Its clear that you feel a lot of love and I wish you the best in life! I hope you stay this way!"
        elif highestEmotion == "neutral":
            finalReply = "You seem to be a person that has their feelings in check, I wish you the best in life!"
        elif highestEmotion == "sad":
            finalReply = "It's okay to be a little sad, just know that a lot of people love and care for you! Reach out to them, you are not alone."
        return finalReply

    def get_response(self, message):
        if message.lower() == "quit":
            outputFromModel = self.finalMessage(self.emotion_gauge)
        else:
            if self.option == "SVM" or self.option == "LOGREG":
                messageCountVector = self.preprocessMessageSVMLOGREG(message)
                predictionScores = self.getPredictionScoresSVMLOGREG(messageCountVector)
                currentMemory = self.updateUserEmotionsSVMLOGREG(predictionScores)
                print("Updated Emotion Dictionary / MEMORY : ", currentMemory)
                print("\n")
                getEmotionDetected = self.getEmotionFeltSVMLOGREG(predictionScores)
                print("Emotion Identified :", getEmotionDetected)
                print("------------------------------------------------------")
                outputFromModel = self.getReply(getEmotionDetected)
            elif self.option == "CNN":
                messagePostPadding = self.preprocessMessageCNN(message)
                predictionScores = self.getPredictionScoresCNN(messagePostPadding)
                currentMemory = self.updateUserEmotionsCNN(predictionScores)
                print("Updated Emotion Dictionary / MEMORY : ", currentMemory)
                print("\n")
                getEmotionDetected = self.getEmotionFeltCNN(predictionScores)
                print("Emotion Identified :", getEmotionDetected)
                print("------------------------------------------------------")
                outputFromModel = self.getReply(getEmotionDetected)
        return outputFromModel
