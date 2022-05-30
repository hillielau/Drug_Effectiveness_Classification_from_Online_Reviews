import torch
import torch.nn as nn

class MLP_Net_layer(nn.Module):
     def __init__(self, input_size, hidden_size, num_classes,activation1):
        super(MLP_Net_layer, self).__init__()
        self.layer_1 = nn.Linear(input_size,hidden_size, bias=True)
        self.activation1 = activation1
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)
 
     def forward(self, x):
        out = self.layer_1(x)
        out = self.activation1(out)
        out = self.output_layer(out)
        return out
