import torch.nn as nn
import torch
from icecream import ic


class TabularModel(nn.Module):
    def __init__(self, params):
        super(TabularModel, self).__init__()

        # self.embedding_categorical = nn.Embedding(params['embedding_size_categorical'], 1)
        embedding_categorical = [
            nn.BatchNorm1d(params['x_shape_categorical']),
            nn.Linear(params['x_shape_categorical'], params['embedding_size_categorical'])
        ]
        self.embedding_categorical = nn.Sequential(*embedding_categorical)
        linear_continuous = [
            nn.BatchNorm1d(params['x_shape_continuous']),
            nn.Linear(params['x_shape_continuous'], params['embedding_size_continuous'])
        ]
        self.linear_continuous = nn.Sequential(*linear_continuous)

        representation_layers = [
            nn.Linear(params['embedding_size_categorical'] + params['embedding_size_continuous'],
                      params['combined_embedding']),
            nn.BatchNorm1d(params['combined_embedding']),
            nn.LeakyReLU(),
            nn.Dropout(params['dropout']),

            nn.Linear(params['combined_embedding'], int(params['combined_embedding']/2)),
            nn.BatchNorm1d(int(params['combined_embedding']/2)),
            nn.LeakyReLU(),
            nn.Dropout(params['dropout']/2),
        ]
        self.representation_layers = nn.Sequential(*representation_layers)

        outcome_layers = [
            nn.Linear(int(params['combined_embedding']/2) + 1, 4),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Dropout(params['dropout']/2),

            nn.Linear(4, 1),
        ]
        self.outcome_layers = nn.Sequential(*outcome_layers)

    def forward(self, x_continuous, x_categorical, t):
        embedding_categorical = self.embedding_categorical(x_categorical)
        embedding_continuous = self.linear_continuous(x_continuous)
        embedding_x = torch.cat([embedding_categorical, embedding_continuous], 1)

        representation_x = self.representation_layers(embedding_x)

        all_factors = torch.cat([representation_x, t], 1)
        predicted_outcome = self.outcome_layers(all_factors)
        predicted_outcome = torch.squeeze(predicted_outcome)

        return representation_x, predicted_outcome
