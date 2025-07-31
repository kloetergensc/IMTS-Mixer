from torch import nn


class MixerBlock(nn.Module):
    def __init__(
        self,
        channels,
        D,
    ):
        super().__init__()
        self.channel_mixer = nn.Sequential(
            nn.RMSNorm([D, channels]),
            nn.Linear(channels, channels),
            nn.ReLU(),
        )
        self.kernel_mixer = nn.Sequential(
            nn.RMSNorm([channels, D]),
            nn.Linear(D, D),
            nn.ReLU(),
        )

    def forward(self, z):
        z = z + self.channel_mixer(z.transpose(1, 2)).transpose(1, 2)
        z = z + self.kernel_mixer(z)
        return z
