import tkinter as tk
import ChatbotGUI as GUI

# Main file which runs the initial option selection screen
class OptionSelection:
    def __init__(self, master):
        self.master = master
        master.title("Companio Version 1.0 - Model Selection")

        master.configure(bg="#7EA0B7")
        master.geometry("400x400")

        label = tk.Label(master, text="Select which AI Model should \nbe used for Emotion Detection", font=("Helvetica", 16))
        label.pack(pady=20)
        SVMButton = tk.Button(master, text="Support Vector Machine", font=("Helvetica", 12), command=lambda: self.runChatbot("SVM"))
        SVMButton.pack(pady=10)
        LogRegButton = tk.Button(master, text="Logistic Regression", font=("Helvetica", 12), command=lambda: self.runChatbot("LOGREG"))
        LogRegButton.pack(pady=10)
        CNNButton = tk.Button(master, text="Convolutional Neural Network", font=("Helvetica", 12), command=lambda: self.runChatbot("CNN"))
        CNNButton.pack(pady=10)

    def runChatbot(self, option):
        self.master.destroy()
        chatbotWindow = tk.Tk()
        GUI.ChatbotGUI(chatbotWindow, option)
        chatbotWindow.mainloop()


if __name__ == "__main__":
    selectionWindow = tk.Tk()
    select = OptionSelection(selectionWindow)
    selectionWindow.mainloop()
