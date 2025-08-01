import torch
import torch.nn as nn

from IMTSMixer.model.iscam import ISCAM
from IMTSMixer.model.mixer_block import MixerBlock


class IMTSMixer_mlp(nn.Module):
    def __init__(
        self,
        channels,
        D,
        D_dec,
        hidden_dim,
        mixer_blocks,
        device,
    ) -> None:
        super().__init__()

        self.out = nn.Sequential(
            nn.Linear(D_dec + 10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.out_kernel = nn.Sequential(
            nn.Linear(1, 10),
        )

        self.iscam = ISCAM(
            channels=channels,
            D=D,
            hidden_dim=hidden_dim,
        )
        self.mixer_out = nn.Sequential(
            nn.RMSNorm([channels, D]),
            nn.Linear(D, D_dec),
        )
        if mixer_blocks > 0:
            self.init_channel_mixer = nn.Sequential(
                nn.RMSNorm([D, channels]),
                nn.Linear(channels, channels),
                nn.ReLU(),
            )
        self.mixer_blocks = nn.ModuleList([MixerBlock(channels=channels, D=D) for i in range(mixer_blocks - 1)])
        self.use_init_mixer = mixer_blocks > 0
        self.device = device

    def forward(self, T, X, M, YT):
        if T.dim() == 2:
            T = T.unsqueeze(-1).expand_as(X)
        z = self.iscam(T=T, X=X, M=M)
        for m in self.mixer_blocks:
            z = z + m(z)
        # BS x C x KD
        if self.use_init_mixer:
            z = z + self.init_channel_mixer(z.transpose(1, 2)).transpose(1, 2)
        z = self.mixer_out(z)
        # yhat: BS x YT x C x KD
        # MLP output
        yhat = z.unsqueeze(1).expand(-1, YT.shape[1], -1, -1)
        query = self.out_kernel(YT.unsqueeze(-1))
        query = torch.cat([query[:, :, 0:1], torch.sin(query[:, :, 1:])], dim=-1)
        query = query.unsqueeze(2).repeat(1, 1, yhat.shape[2], 1)
        return self.out(torch.cat([yhat, query], dim=-1)).squeeze(-1)

    def forecasting_with_targets(self, batch_dict):
        X = batch_dict["observed_data"].to(self.device)
        T = batch_dict["observed_tp"].to(self.device)
        M = batch_dict["observed_mask"].to(torch.bool).to(self.device)
        Y = batch_dict["data_to_predict"].to(self.device)
        YT = batch_dict["tp_to_predict"].to(self.device)
        MY = batch_dict["mask_predicted_data"].to(self.device)
        YHAT = self.forward(T=T, X=X, M=M, YT=YT)
        return YHAT, Y, MY
