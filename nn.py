import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Word2VecNetwork(nn.Module):
    # embedding_dimension -> hidden layer dimension
    def __init__(self, input_dimension, embedding_dimension):
        super(Word2VecNetwork, self).__init__()

        self.input_dimension = input_dimension
        self.embedding_dimension = embedding_dimension

        # define layers
        self.layer1 = nn.Linear(input_dimension, embedding_dimension)
        self.layer2 = nn.Linear(embedding_dimension, input_dimension)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = F.softmax(out, dim=0)

        return out


"""
Trains the neural network
optimizer : SGD
loss : cross entropy, or you can use negative log likelihood by adding an extra layer after the final
        layer to the log likelihood, as mentioned in - 
        https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
"""


def train(input_dimension, embedding_dimension, learning_rate, focus_words, contexts, epochs=20000):
    # init device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Word2VecNetwork(input_dimension, embedding_dimension)

    print("Model Description")
    print("=================")
    print(model)
    print("=================")
    print(f"Device : {device}")
    print()

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

    return model
