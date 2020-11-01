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
def who_is_wayne(m):
    w1 = model.layer1.weight
    # stack to merge rows like row_1 -> col_1 with row_2 -> col_2 as a single row
    # print w1 in case you need an explanation!
    w1 = torch.stack((w1[0], w1[1]), dim=1)

    b1 = model.layer1.bias

    # word vectors
    vectors = w1 + b1

    print("")
    print("Preparing vectors")

    # cosine similarity to check how similar two vectors are
    cos = nn.CosineSimilarity(dim=0)

    bat_wayne = cos(vectors[p.int_labels["wayne"]], vectors[p.int_labels["batman"]])
    joker_wayne = cos(vectors[p.int_labels["wayne"]], vectors[p.int_labels["joker"]])

    identity = "batman" if bat_wayne > joker_wayne else "joker"
    print("======================================")
    print(f"The Skynet says : Wayne is {identity}")


who_is_wayne(model)
