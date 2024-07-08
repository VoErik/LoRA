import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank, alpha):
        super().__init__()
        self.alpha = alpha
        self.rank = rank
        self.scaling = self.alpha / self.rank
        self.A = torch.nn.Parameter(torch.zeros(input_dim, self.rank))
        self.B = torch.nn.Parameter(torch.zeros(rank, output_dim))
        nn.init.normal_(self.A, mean=0, std=1)

    def forward(self, x):
        x = self.scaling * (x @ self.A @ self.B)
        return x


class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def replace_modules_with_lora(model, modules_to_replace, rank, alpha):
    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.named_children():
        if name in modules_to_replace:
            new_module = LoRALinear(module, rank, alpha)
            setattr(model, name, new_module)
        elif len(list(module.children())) > 0:
            replace_modules_with_lora(module, modules_to_replace, rank, alpha)  # Recursively replace in child modules

    return model


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]

        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
