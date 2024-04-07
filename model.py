import torch.nn as nn


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class=2):
        super().__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU() # функция активации
        self.fc2 = nn.Linear(embed_dim, num_class)

    def forward(self, batch_voc, offsets):
        embedded = self.embedding(batch_voc, offsets)

        x = self.fc1(embedded)
        y = self.relu(x)
        z = self.fc2(y)

        return z
