import torch
import torchvision.transforms as T
from PIL import Image
from dataclasses import dataclass
import numpy as np

@dataclass
class StateProcessor:
    transformer: T.Compose = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.CenterCrop((160, 160)),
        T.Resize((84, 84), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor()
    ])

    def process_state(self, state) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            state = T.ToPILImage()(state)
        elif isinstance(state, np.ndarray):
            state = Image.fromarray(state)

        # state = self.transformer(state)

        return self.transformer(state).unsqueeze(0)