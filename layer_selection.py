import torch.nn as nn
import base_layers
import inspect
from functools import partial

class SelecModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        self.architecture_dict = {0:"None"}
        functions_list = inspect.getmembers(base_layers, inspect.isfunction)

        for i, func in enumerate(functions_list):
            self.architecture_dict[i+1] = {"name":func[0],
                                           "func":func[1]}
            
        self.outputs_inst = nn.ModuleList()
        self.outputs = partial(nn.Linear, out_features=output_features)
        # print(self.outputs)

        self.selection = None
        spaces = []
        for i, layer in enumerate(self.architecture_dict.values()):
            print(layer)
        # for n, layer in enumerate(self.base_layers):
        #     spaces.append(layer.space)
            
    def sample(self):
        # Copy the functionality from the spaces.py file
        sample = []
        for space in self.spaces:
            if isinstance(space, ChooseSpace):
                i = space.sample()
                sample.append(i)
                space = space.spaces[i]
            sample.extend(space.sample())
        return sample
    
    def select(self, selection):
        # Used for selecting a architecture space after sampling
        self.selection = selection
        return selection
    
    def forward(self, inputs):
        selection = self.selection
        layer_input = inputs
        list_layer_input = [layer_input]
        activations = [layer_input]
        layer_output = [layer_input]

#Used for testing remove later
if __name__ == "__main__":
    modelselect = SelecModel(10)
