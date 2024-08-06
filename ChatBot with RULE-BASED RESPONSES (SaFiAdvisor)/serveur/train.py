import json
from index import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

class ChatDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

if __name__ == "__main__":
    # Open the JSON file with explicit UTF-8 encoding
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)

    all_words = []  # store all the words
    tags = []  # store tags
    tagsandpatterns = []

    for intent in intents['intents']:
        tag = intent['tag']  # collect tags
        tags.append(tag)  # add every tag collected to the list of tags
        for pattern in intent['patterns']:
            w = tokenize(pattern)  # tokenize the pattern
            all_words.extend(w)  # add each word of the tokenized word to the all_words list
            tagsandpatterns.append((w, tag))  # add the tag and the tokenized word to the tag and patterns list

    ignore_words = ['?', '!', '.', ',']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    x_train = []  # words
    y_train = []  # tags

    for (pattern_sentence, tag) in tagsandpatterns:
        bag = bag_of_words(pattern_sentence, all_words)
        x_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)  # CrossEntropyLoss

    x_train = np.array(x_train)
    y_train = np.array(y_train)

   

    # Hyperparameters
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(x_train[0])
    learning_rate = 0.001
    num_epochs = 1000

    dataset = ChatDataset(x_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers=0 for debugging

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (words, labels) in enumerate(train_loader):
            words = words.to(device)
            labels = labels.to(device, dtype=torch.long)  # Ensure labels are of type LongTensor

            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}')

    print(f'final loss, loss = {loss.item():.4f}')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'training complete. File saved to {FILE}')
