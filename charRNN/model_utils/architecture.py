import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_size=128,
        hidden_size=256,
        n_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.encoder = nn.Embedding(vocabulary_size, embedding_size)
        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, x):
        x = self.encoder(x)
        x, _ = self.rnn(x)
        # X tiene dimensiones (N,L,D)
        x = self.fc(x[:, -1, :])
        return x
