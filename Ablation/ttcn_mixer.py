import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Encoder(nn.Module):
    def __init__(
        self,
        channels,
        D,
        D_dec,
        kernel_hidden_dim,
        mixer_blocks=2,
    ):
        super().__init__()
        input_dim = 11
        self.Filter_Generators = nn.Sequential(
            nn.Linear(input_dim, D, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(D, D, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(D, input_dim * D, bias=True),
        )

        self.channels = channels
        self.out_layer = nn.Sequential(
            nn.RMSNorm([channels, D]),
            nn.Linear(D, D_dec),
        )
        if mixer_blocks > 0:
            self.init_channel_mixer = nn.Sequential(
                nn.RMSNorm([D, channels]),
                nn.Linear(channels, channels),
                nn.ReLU(),
            )
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(channels=channels, D=D) for i in range(mixer_blocks - 1)]
        )
        self.use_init_mixer = mixer_blocks > 0
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, 9)
        self.ttcn_dim = D
        self.T_bias = nn.Parameter(torch.randn(1, D))

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def TTCN(self, X_int, mask_X):
        # X_int: shape (B*N*M, T, F)
        # mask_X: shape (B*N*M, T, 1)

        N, Lx, _ = mask_X.shape
        Filter = self.Filter_Generators(X_int)  # (N, Lx, F_in*ttcn_dim)
        Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)
        # normalize along with sequence dimension
        Filter_seqnorm = F.softmax(Filter_mask, dim=-2)  # (N, Lx, F_in*ttcn_dim)
        Filter_seqnorm = Filter_seqnorm.view(
            N, Lx, self.ttcn_dim, -1
        )  # (N, Lx, ttcn_dim, F_in)
        X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)
        ttcn_out = torch.sum(
            torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1
        )  # (N, ttcn_dim)
        h_t = torch.relu(ttcn_out + self.T_bias)  # (N, ttcn_dim)
        return h_t

    def forward(self, T, X, M):
        TE = self.LearnableTE(T.unsqueeze(-1))
        XT = torch.cat([X.unsqueeze(-1), TE], dim=-1).permute(0, 2, 1, 3)
        z = self.TTCN(
            XT.flatten(start_dim=0, end_dim=1),
            M.permute(0, 2, 1).unsqueeze(-1).flatten(start_dim=0, end_dim=1).to(int),
        )
        z = z.view(X.shape[0], X.shape[2], -1)
        # BS x T x C x D
        for m in self.mixer_blocks:
            z = z + m(z)
        # BS x C x KD
        if self.use_init_mixer:
            z = z + self.init_channel_mixer(z.transpose(1, 2)).transpose(1, 2)
        z = self.out_layer(z)
        return z


class TTCN_Mixer(nn.Module):
    def __init__(
        self,
        channels,
        D,
        D_dec,
        kernel_hidden_dim,
        mixer_blocks,
        device,
    ) -> None:
        super().__init__()
        self.out_kernel = nn.Sequential(
            nn.Linear(1, kernel_hidden_dim),
            nn.ReLU(),
            nn.Linear(kernel_hidden_dim, D_dec * channels),
        )
        self.out = nn.Sequential(
            nn.Linear(D_dec, 1),
        )

        self.encoder = Encoder(
            channels=channels,
            D=D,
            D_dec=D_dec,
            kernel_hidden_dim=kernel_hidden_dim,
            mixer_blocks=mixer_blocks,
        )
        self.device = device

    def forward(self, T, X, M, YT, MY):
        # X_enc: BS x T xC x KD
        if T.dim() == 2:
            T = T.unsqueeze(-1).expand_as(X)
        z = self.encoder(T=T, X=X, M=M)
        # yhat: BS x YT x C x KD
        yhat = z.unsqueeze(1).expand(-1, YT.shape[1], -1, -1)
        query = self.out_kernel(YT.unsqueeze(-1)).unsqueeze(-1).reshape_as(yhat)

        return self.out(yhat * query).squeeze(-1)

    def forecasting_with_targets(self, batch_dict):
        X = batch_dict["observed_data"].to(self.device)
        T = batch_dict["observed_tp"].to(self.device)
        M = batch_dict["observed_mask"].to(torch.bool).to(self.device)
        Y = batch_dict["data_to_predict"].to(self.device)
        YT = batch_dict["tp_to_predict"].to(self.device)
        MY = batch_dict["mask_predicted_data"].to(self.device)
        YHAT = self.forward(T=T, X=X, M=M, YT=YT, MY=MY)
        return YHAT, Y, MY
