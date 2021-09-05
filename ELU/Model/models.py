import torch


class CNNClassifier(torch.nn.Module):
    # HIGH SCORE: 81.6
    # layers = [32, 64, 128]
    # conv2d (ks7, p3, s2), relu
    # conv2d (ks3, p1, s2), relu, conv2d (ks3, p1, s1), relu
    class CNNBlock(torch.nn.Module):
        def __init__(self, n_input: int, n_output: int, batch_norm: bool, activation_function):
            super().__init__()
            if batch_norm:
                self.net = torch.nn.Sequential(
                    torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding = 1, stride=2),
                    torch.nn.BatchNorm2d(n_output),
                    activation_function(),
                    torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding = 1, stride=2),
                    torch.nn.BatchNorm2d(n_output),
                    activation_function(),
                )
            else:
                self.net = torch.nn.Sequential(
                    torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=2),
                    activation_function(),
                    torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding = 1, stride=2),
                    activation_function()
                )
        
        def forward(self, x):
            return self.net(x)            

    def __init__(self, activation_function=torch.nn.ReLU, n_input_channels: int = 3, batch_norm=False):
        #[1 ×192 ×5],[1 ×192 ×1,1 ×240 ×3],[1 ×240 ×1,1 ×260 ×2],[1 ×260 ×1,1 ×280 ×2],[1 × 280 × 1,1 × 300 × 2],[1 × 300 × 1],[1 × 100 × 1]
        super().__init__()
        if batch_norm:
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, kernel_size=5, padding=2, stride=2),
                torch.nn.BatchNorm2d(32),
                activation_function(),

                self.CNNBlock(32, 64, batch_norm, activation_function),
                self.CNNBlock(64, 96, batch_norm, activation_function),
                torch.nn.Conv2d(96, 128, kernel_size=3, padding=1, stride=2),
                activation_function()
            )
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, kernel_size=5, padding=2, stride=2),
                activation_function(),

                self.CNNBlock(32, 64, batch_norm, activation_function),
                self.CNNBlock(64, 96, batch_norm, activation_function),
                torch.nn.Conv2d(96, 128, kernel_size=3, padding=1, stride=2),
                activation_function()
            )
        self.classifier = torch.nn.Linear(128, 10)
    
    def forward(self, x):
        # Normalize the values of the images
        x[:, 0] = (x[:, 0] - 0.5) / 0.5
        x[:, 1] = (x[:, 1] - 0.5) / 0.5
        x[:, 2] = (x[:, 2] - 0.5) / 0.5
        z = self.net(x)
        z = z.mean(dim=[2,3])

        # Classify
        return self.classifier(z)
        


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r