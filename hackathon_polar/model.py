import torch
import torch.nn as nn

class _SepConv1d(nn.Module):
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SepConv1d(nn.Module):
    def __init__(self, ni, no, kernel, stride, pad, drop=None,
                 activ=lambda: nn.ReLU(inplace=True)):
    
        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.layers(x)
      

class Flatten(nn.Module):
    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)

class Classifier(nn.Module):
    def __init__(self, raw_ni, no, drop=.5):
        super().__init__()
        
        self.elev_process = nn.Sequential(
            # in_channel, out_channel, kernel_size, stride, padding
            SepConv1d( raw_ni, 64, 3, 1, 0, drop=drop),
            SepConv1d( 64,  128, 3, 1, 0, drop=drop),
            SepConv1d( 128,  64, 3, 1, 0, drop=drop),
            SepConv1d( 64,  8, 3, 1, 0, drop=drop),
            nn.MaxPool1d(3, stride=3),
            Flatten(),
            nn.Linear(248, 16), nn.ReLU(inplace=True),
            )

        self.out = nn.Linear(16 + 1, no)      
        
    def forward(self, elev, hydpot):
        elev_process_out = self.elev_process(elev)
        hydpot = hydpot.unsqueeze(1)
        concated = torch.cat((elev_process_out, hydpot), dim=1)
        return self.out(concated)
