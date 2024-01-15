# from base_layers import conv1x1
import base_layers
import inspect
import random
print(dir(base_layers))

# Get all functions from my_module
functions_list = inspect.getmembers(base_layers, inspect.isfunction)

for func in functions_list:
    print(f"Function name: {func[0]}, Function object: {func[1]}")

architecture_indices = {0:"None"}

for i, func in enumerate(functions_list):
    architecture_indices[i+1] = func[0]

print(architecture_indices)

print( random.choice(list(architecture_indices.values())))