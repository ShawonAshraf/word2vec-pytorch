from preprocessing import Preprocessor
from nn import train
import torch
import torch.nn as nn

p = Preprocessor(dimensions=2)
data, focus_words, contexts = p.run()

model = train(
    input_dimension=len(p.vocabulary),
    embedding_dimension=2,
    learning_rate=0.1,
    focus_words=focus_words,
    contexts=contexts,
    tr=False
)

# check for similarity between batman and wayne vs joker and wayne
w1 = model.layer1.weight
# stack to merge rows on dim=1 (by column)
w1 = torch.stack((w1[0], w1[1]), dim=1)

b1 = model.layer1.bias

vectors = w1 + b1

cos = nn.CosineSimilarity(dim=0)

bat_wayne = cos(vectors[p.int_labels["american"]], vectors[p.int_labels["batman"]])
print(bat_wayne)
joker_wayne = cos(vectors[p.int_labels["wayne"]], vectors[p.int_labels["joker"]])
print(joker_wayne)

print(bat_wayne > joker_wayne)
