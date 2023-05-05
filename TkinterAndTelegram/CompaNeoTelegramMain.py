import API as Keys
from telegram.ext import *
import SVM as SVM
import LogReg as LOGREG
import HNN as HNN

updater = Updater(Keys.API_KEY)
optionChosen = []

def startBot(update, param):
    name = update.message.chat.first_name
    update.message.reply_text('Hi ' + name + '!')
    update.message.reply_text('I am CompaNeo')
    update.message.reply_text('Type /svm to load my Support Vector Machine Emotion Detection')
    update.message.reply_text('Type /logreg to load my Logistic Regression Emotion Detection')
    update.message.reply_text('Type /hnn to load my Hybrid Neural Network Emotion Detection')

def botHelp(update, param):
    update.message.reply_text('You can use /start to begin')
    update.message.reply_text("You can type 'quit' to end")

def getUserMessage(update, param):
    message = str(update.message.text).lower()
    if message == "/svm":
        optionChosen.append("SVM")
        response = SVM.get_response(message)
    elif message == "/logreg":
        optionChosen.append("LOGREG")
        response = LOGREG.get_response(message)
    elif message == "/hnn":
        optionChosen.append("HNN")
        response = HNN.get_response(message)
    elif optionChosen[-1] == "SVM":
        response = SVM.get_response(message)
    elif optionChosen[-1] == "LOGREG":
        response = LOGREG.get_response(message)
    elif optionChosen[-1] == "HNN":
        response = HNN.get_response(message)
    else:
        response = "Please load in a model for us to converse!"
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