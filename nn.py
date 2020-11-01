import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Word2VecNetwork(nn.Module):
    # embedding_dimension -> hidden layer dimension
    def __init__(self, input_dimension, embedding_dimension, epochs, learning_rate):
        super(Word2VecNetwork, self).__init__()

        self.input_dimension = input_dimension
        self.embedding_dimension = embedding_dimension

        self.epochs = epochs
        self.learning_rate = learning_rate

        # define layers
        self.layer1 = nn.Linear(input_dimension, embedding_dimension)
        self.layer2 = nn.Linear(embedding_dimension, input_dimension)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = F.softmax(out, dim=0)

        return out


def train(input_dimension, embedding_dimension, learning_rate, focus_words, contexts, epochs=20000, tr=True):
    # init device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Word2VecNetwork(input_dimension, embedding_dimension, epochs, learning_rate)

    print("Model Description")
    print("=================")
    print(model)
    print("=================")
    print()

    # if the model already exists on disk and tr is false, just return it!
    if os.path.exists("model.ckpt") and tr is False:
        model.load_state_dict(torch.load("model.ckpt"))
        return model

    # loss and optimization
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        X = torch.from_numpy(focus_words).to(device)
        y = torch.from_numpy(contexts).to(device)

        # forward pass
        outputs = model(X)
        # check https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/5
        loss = criterion(outputs, torch.max(y, 1)[1])

        # backprop
        optim.zero_grad()
        loss.backward()
        optim.step()

        # print loss for every 3000 steps
        if e % 3000 == 0:
            print(f"epoch = [{e}/{epochs}] # loss = {loss.item()}")

    # save the model
    torch.save(model.state_dict(), "model.ckpt")

    return model
