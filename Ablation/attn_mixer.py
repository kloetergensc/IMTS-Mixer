import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from IMTSMixer.model.mixer_block import MixerBlock


class MAB2(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, n_dim, num_heads, ln=False):
        super(MAB2, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.n_dim = n_dim
        self.fc_q = nn.Linear(dim_Q, n_dim)
        self.fc_k = nn.Linear(dim_K, n_dim)
        self.fc_v = nn.Linear(dim_K, n_dim)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(n_dim, n_dim)

    def forward(self, Q, K, mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.n_dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, -1), 0)
        K = torch.cat(K.split(dim_split, -1), 0)
        V = torch.cat(V.split(dim_split, -1), 0)

        Att_mat = Q_.matmul(K.transpose(-1, -2)) / math.sqrt(self.n_dim)
        if mask is not None:
            Att_mat = Att_mat.masked_fill(mask.repeat(self.num_heads, 1, 1, 1) == 0, -10e9)
        A = torch.softmax(Att_mat, -1)
        O = torch.cat((Q_ + A.matmul(V)).split(Q.size(0), 0), -1)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class Encoder(nn.Module):
    def __init__(
        self,
        channels,
        D,
        n_heads,
    ):
        super().__init__()

        self.channels = channels
        self.channel_time_attn = MAB2(dim_Q=D, dim_K=D, dim_V=D, n_dim=D, num_heads=n_heads)
        self.x_embed = nn.Linear(1, D // 2)
        self.te_periodic = nn.Linear(1, (D // 2) - 1)
        self.te_scale = nn.Linear(1, 1)
        self.channel_emb = nn.Parameter(torch.randn(channels, D))

    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def forward(self, T, X, M):
        TE = self.LearnableTE(T.unsqueeze(-1))
        X_emb = self.x_embed(X.unsqueeze(-1))
        XT = torch.cat([X_emb, TE], dim=-1).permute(0, 2, 1, 3)
        C_emb = self.channel_emb.unsqueeze(0).unsqueeze(2).repeat(X.size(0), 1, 1, 1)
        attn_mask = M.permute(0, 2, 1).unsqueeze(-2)
        z = self.channel_time_attn(Q=C_emb, K=XT, mask=attn_mask)
        z = z.squeeze(-2)
        # BS x T x C x D
        return z


class Attn_Mixer(nn.Module):
    def __init__(
        self,
        channels,
        D,
        D_dec,
        hidden_dim,
        mixer_blocks,
        n_heads,
        device,
    ) -> None:
        super().__init__()
        self.out_kernel = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, D_dec * channels),
        )
        self.out = nn.Sequential(
            nn.Linear(D_dec, 1),
        )

        self.mha_encoder = Encoder(
            channels=channels,
            D=D,
            n_heads=n_heads,
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
        z = self.mha_encoder(T=T, X=X, M=M)
        for m in self.mixer_blocks:
            z = z + m(z)
        # BS x C x KD
        if self.use_init_mixer:
            z = z + self.init_channel_mixer(z.transpose(1, 2)).transpose(1, 2)
        z = self.mixer_out(z)
        # yhat: BS x YT x C x KD
        # ConTP
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
        YHAT = self.forward(T=T, X=X, M=M, YT=YT)
        return YHAT, Y, MY
