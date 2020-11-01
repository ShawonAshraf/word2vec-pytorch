# word2vec-pytorch
A simple, barebone, basic, newb implementation of word2vec using pytorch.

## neural network
- SGD as Optimizer
- 20K epochs (more will reduce the error but the dataset here is smaller than baby Yoda.)
- Learning rate : 0.1
- Cross Entropy as Loss function

## run
First set up a ``conda`` environment using the file `snek.yml`. Then run `main.py`.

```bash
python main.py
```

## what to expect
```bash
Model Description
=================
Word2VecNetwork(
  (layer1): Linear(in_features=19, out_features=2, bias=True)
  (layer2): Linear(in_features=2, out_features=19, bias=True)
)
=================
Device : cpu

Model already exists! Skipping training

Preparing vectors
======================================
The Skynet says : Wayne is batman
```

## license?
You must be fun at parties, no?