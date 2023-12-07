#!/bin/env python

from sentence_transformers import SentenceTransformer, models, InputExample, losses
from torch.utils.data import DataLoader
import csv

word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_examples = []

with open('data/training.tsv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter = '\t')
    for row in csvreader:
        # Each row is a list of values
        ex = InputExample(texts = [row[1], row[2]], label = float(row[-1]))
        train_examples.append(ex)

#Define your train examples. You need more than just two examples...
# train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
#     InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]

#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=3)
train_loss = losses.CosineSimilarityLoss(model)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)