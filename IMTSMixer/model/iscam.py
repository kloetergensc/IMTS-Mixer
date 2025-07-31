import torch
from torch import nn


class Tuple_Encoder(nn.Module):

    def __init__(
        self,
        hd,
        D,
    ):
        super().__init__()

        self.nn = nn.Sequential(nn.Linear(2, hd), nn.ReLU(), nn.Linear(hd, D))

    def forward(self, X, T):
        return self.nn(torch.cat([X.unsqueeze(-1), T.unsqueeze(-1)], -1))


class ISCAM(nn.Module):
    def __init__(
        self,
        channels,
        D,
        hidden_dim,
    ):
        super().__init__()
        self.tuple_encoderH = Tuple_Encoder(
            hd=hidden_dim,
            D=D,
        )
        self.tuple_encoderW = Tuple_Encoder(
            hd=hidden_dim,
            D=D,
        )
        self.channel_bias = nn.Parameter(torch.randn(channels, D))
        self.channels = channels

    def forward(self, T, X, M):
        # BS x T x C x KD
        H = self.tuple_encoderH(X=X, T=T)
        W = self.tuple_encoderW(X=X, T=T)
        W = W - (~M.unsqueeze(-1) * 10e9)
        W = W.softmax(dim=1)
        channel_mask = M.sum(1) > 0
        # z: BS x C x KD
        aggregated_channels = ((H * W).sum(1)) * channel_mask.unsqueeze(-1)
        z = torch.zeros_like(aggregated_channels)
        z += self.channel_bias.unsqueeze(0)
        z += aggregated_channels
        return z
