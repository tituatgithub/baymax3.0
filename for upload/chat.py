import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu') #to run on gpu 

with open('intents.json', 'r') as f:
    intents = json.load(f)
File= 'data.pth'
data= torch.load(File, weights_only=True)  # Set weights_only=True to avoid the warning

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Baymax_3.0"

def get_responce(msg):
    sentence = msg
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "The server is busy. Please try again later..."



