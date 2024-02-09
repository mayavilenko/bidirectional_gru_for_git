import torch
from GRU_model import GRUModel
import torch.nn as nn
import numpy as np
import random


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())
        
        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out


torch.manual_seed(1)
np.random.seed(2)
random.seed(3)

SequenceLength = 13
Features = 1
OutputDim = 1
HiddenSize = 64
LayersDim = 1
DropoutProb = 0.0
#Lr = 0.001
Lr = 0.09138694552832571 #post-optuna
Criterion = nn.MSELoss()
Epochs = 500
BatchSize = 32
#Year = 2020

Criterion = nn.MSELoss()
TbDirectory='tbs/'
#Define our device
Device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#alpha=np.exp(-5)
Model=GRUModel(input_dim=Features, hidden_dim=HiddenSize, layer_dim=LayersDim, output_dim=OutputDim, dropout_prob=DropoutProb)
Optimizer=torch.optim.AdamW(Model.parameters(), lr=Lr)

