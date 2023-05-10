import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime
import pickle
import json
import random
import nltk
from nltk.corpus import stopwords
from textblob import Word
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
nltk.download('wordnet')

# Class which runs the initial option selection screen
class OptionSelection:
    def __init__(self, master):
        self.master = master
        master.title("CompaNeo Version 1.0 - Model Selection")
        master.configure(bg="#7EA0B7")
        master.geometry("400x400")

        label = tk.Label(master, text="Select which AI Model should \nbe used for Emotion Detection",
                         font=("Helvetica", 16))
        label.pack(pady=20)
        SVMButton = tk.Button(master, text="Support Vector Machine", font=("Helvetica", 12),
                              command=lambda: self.runChatbot("SVM"))
        SVMButton.pack(pady=10)
        LogRegButton = tk.Button(master, text="Logistic Regression", font=("Helvetica", 12),
                                 command=lambda: self.runChatbot("LOGREG"))
        LogRegButton.pack(pady=10)
        HNNButton = tk.Button(master, text="Hybrid Neural Network", font=("Helvetica", 12),
                              command=lambda: self.runChatbot("HNN"))
        HNNButton.pack(pady=10)

    def runChatbot(self, option):
        self.master.destroy()
        chatbotWindow = tk.Tk()
        ChatbotGUI(chatbotWindow, option)
        chatbotWindow.mainloop()

# Class which handles all GUI components of the chatbot
class ChatbotGUI:
    def __init__(self, master, option):
        self.master = master
        master.title("CompaNeo Version 1.0 - Chat Section")

        self.chat_tags = {
            "user": {
                "bg": "#5aa57c",
                "font": "#000000",
            },
            "companio": {
                "bg": "#a0acc2",
                "font": "#000000",
            },
        }

        self.chatHistory = tk.scrolledtext.ScrolledText(master, width=50, height=30, wrap=tk.WORD,
                                                        font=("Helvetica", 12))
        self.chatHistory.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="NSEW")
        self.inputChatBox = tk.Entry(master, width=50, font=("Helvetica", 12))
        self.inputChatBox.grid(row=1, column=0, padx=10, pady=10, sticky="EW")
        self.inputChatBox.bind("<Return>", self.sendMessage)
        self.sendButton = tk.Button(master, text="Send", font=("Helvetica", 12), command=self.sendMessage, bg="#228CDB")
        self.sendButton.grid(row=1, column=1, padx=10, pady=10, sticky="E")
        master.grid_columnconfigure(0, weight=1)
        self.chatHistory.config(state=tk.DISABLED)
        self.chatbot = Chatbot(option)
        self.addMessage(" Selected Algorithm: " + option)

    def sendMessage(self, event=None):
        message = self.inputChatBox.get()
        self.inputChatBox.delete(0, tk.END)
        self.addBubbleMessage("\n  You (" + datetime.now().strftime("%H:%M:%S") + ") :" + message, "user")
        response = self.chatbot.get_response(message)
        self.addBubbleMessage("\n  CompaNeo :" + response, "companio")

    def addBubbleMessage(self, message, tag):
        self.chatHistory.config(state=tk.NORMAL)
        self.chatHistory.tag_configure(tag, background=self.chat_tags[tag]["bg"],
                                       foreground=self.chat_tags[tag]["font"])
        bubble_text = f"{message}"
        bubble_text = f"{bubble_text}\n"
        self.chatHistory.insert(tk.END, bubble_text, (tag, self.chat_tags[tag]))
        self.chatHistory.config(state=tk.DISABLED)
        self.chatHistory.yview(tk.END)

    def addMessage(self, message):
        self.chatHistory.config(state=tk.NORMAL)
        self.chatHistory.insert(tk.END, message + "\n")
        self.chatHistory.config(state=tk.DISABLED)
        self.chatHistory.yview(tk.END)


# Class which contains all the logic of the chatbot
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
        self.emotionsHNN = ["neutral", "happy", "sad", "love", "anger"]
        self.emotion_gauge = {"anger": 0, "happy": 0, "love": 0, "neutral": 0, "sad": 0}
        self.responsesJSON = json.load(open('responses.json', encoding="utf8"))

        if option == "SVM":
            with open('SVMWeights', 'rb') as SVMFiles:
                svm_model, CVectorizer = pickle.load(SVMFiles)
                self.vectorizer = CVectorizer
                self.model = svm_model
        elif option == "LOGREG":
            with open('LogRegWeights', 'rb') as LogRegFiles:
                logreg, CVectorizer = pickle.load(LogRegFiles)
                self.vectorizer = CVectorizer
                self.model = logreg
        elif option == "HNN":
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            self.model = load_model("HNNWeights.h5")

    def preprocessMessageSVMLOGREG(self, message):
        message = ' '.join([word for word in message.split() if word not in self.stop])
        message = " ".join([Word(word).lemmatize() for word in message.split()])
        messageCountVector = self.vectorizer.transform([message])
        return messageCountVector

    def preprocessMessageHNN(self, message):
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

    def getPredictionScoresHNN(self, messagePostPadding):
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

    def updateUserEmotionsHNN(self, predictionScores):
        # In HNN - neutral, happy, sad, love, anger
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

    def getEmotionFeltHNN(self, predictionScores):
        highest_score = max(predictionScores)
        prediction_list = list(predictionScores)
        indexOfMaxEmotion = prediction_list.index(highest_score)
        emotion_felt = self.emotionsHNN[indexOfMaxEmotion]
        return emotion_felt

    def getReply(self, emotion_felt):
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
            if self.responsesJSON['responses_outer'][i]['emotion'] == emotion_felt:
                replySuffix = str(self.responsesJSON['responses_outer'][i]['responses'][
                                      random.randrange(0, len(self.responsesJSON['responses_outer'][i]['responses']))])
        reply = replyPrefix + "  CompaNeo: " + replySuffix
        return reply

    @staticmethod
    def finalMessage(emotion_gauge):
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

    def get_response(self, message):
        if message.lower() == "/quit":
            outputFromModel = self.finalMessage(self.emotion_gauge)
            self.emotion_gauge = {"anger": 0, "happy": 0, "love": 0, "neutral": 0, "sad": 0}
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
            elif self.option == "HNN":
                messagePostPadding = self.preprocessMessageHNN(message)
                predictionScores = self.getPredictionScoresHNN(messagePostPadding)
                currentMemory = self.updateUserEmotionsHNN(predictionScores)
                print("Updated Emotion Dictionary / MEMORY : ", currentMemory)
                print("\n")
                getEmotionDetected = self.getEmotionFeltHNN(predictionScores)
                print("Emotion Identified :", getEmotionDetected)
                print("------------------------------------------------------")
                outputFromModel = self.getReply(getEmotionDetected)
        return outputFromModel


# Run it all
root = tk.Tk()
options = OptionSelection(root)
root.mainloop()