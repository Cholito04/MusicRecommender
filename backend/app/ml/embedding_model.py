import torch.nn as nn


# simple feedforward network to map song features into embedding space
class SimpleEmbeddingModel(nn.Module):
    def __init__(self, input_size=9, hidden_sizes=[8, 6], embedding_size=3):
        super().__init__()
        layers = [
            nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], embedding_size)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
