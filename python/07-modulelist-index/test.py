import torch
from typing import Any, List, Tuple

@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

class Model(torch.jit.ScriptModule):
    def __init__(self, ):
        super(Model, self).__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(2, 2), torch.nn.Softmax(1)
        ])

    @torch.jit.script_method
    def forward(self, input, index: int):
        value: ModuleInterface = self.layers[index]
        return value.forward(input)

print(torch.__version__)
model = Model()
model(torch.randn(2, 2), 1)
model(torch.randn(2, 2), 0)
