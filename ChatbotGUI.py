import tkinter as tk
from tkinter import scrolledtext
import Chatbot as chatBot

# Contains all GUI components of the Chatbot Screen
class ChatbotGUI:
    def __init__(self, master, option):
        self.master = master
        master.title("Companio Version 1.0 - Chat Section")

        self.chatHistory = tk.scrolledtext.ScrolledText(master, width=50, height=20, wrap=tk.WORD,
                                                        font=("Helvetica", 12))
        self.chatHistory.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="NSEW")

        self.inputChatBox = tk.Entry(master, width=50, font=("Helvetica", 12))
        self.inputChatBox.grid(row=1, column=0, padx=10, pady=10, sticky="EW")
        self.inputChatBox.bind("<Return>", self.sendMessage)

        self.sendButton = tk.Button(master, text="Send", font=("Helvetica", 12), command=self.sendMessage, bg="#228CDB")
        self.sendButton.grid(row=1, column=1, padx=10, pady=10, sticky="E")

        master.grid_columnconfigure(0, weight=1)

        self.chatHistory.config(state=tk.DISABLED)
        self.chatbot = chatBot.Chatbot(option)
        self.addMessage("Selected option: " + option)

    def sendMessage(self, event=None):
        message = self.inputChatBox.get()
        self.inputChatBox.delete(0, tk.END)

        self.addMessage("You: " + message)
        response = self.chatbot.get_response(message)
        self.addMessage("Chatbot: " + response)

    def addMessage(self, message):
        self.chatHistory.config(state=tk.NORMAL)
        self.chatHistory.insert(tk.END, message + "\n")
        self.chatHistory.config(state=tk.DISABLED)
        self.chatHistory.yview(tk.END)
