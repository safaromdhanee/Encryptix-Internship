import random
import json
import torch 
from model import NeuralNet
from index import tokenize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)
# read the data from the intents.json 

FILE = "data.pth"
data = torch.load(FILE)
# Load model data
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "SaFiAdvisor"
print("let's chat! type 'quit' to exit")
while True:
    sentence = input('you: ')
    if sentence == 'quit':
        break
    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(device)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand ...")
