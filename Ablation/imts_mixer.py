import math

import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_dim,
        dropout,
    ):
        super().__init__()
        self.channel_mixer = nn.Sequential(
            nn.RMSNorm([hidden_dim, channels]),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.kernel_mixer = nn.Sequential(
            nn.RMSNorm([channels, hidden_dim]),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, z):
        # z: BS x C x D
        z = z + self.channel_mixer(z.transpose(1, 2)).transpose(1, 2)
        z = z + self.kernel_mixer(z)
        return z


class FC_mixer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_dim,
        dropout,
    ):
        super().__init__()
        self.fc_mixer = nn.Sequential(
            nn.Linear(channels * hidden_dim, channels * hidden_dim),
            nn.ReLU(),
        )

    def forward(self, z):
        # z: BS x C x D
        z_old = z.clone()
        z = self.fc_mixer(z.flatten(start_dim=1)).unsqueeze(-1).reshape_as(z_old)
        return z


class multiTimeAttention(nn.Module):

    def __init__(self, embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.linears = nn.ModuleList(
            [nn.Linear(embed_time, embed_time), nn.Linear(embed_time, embed_time)]
        )

    def attention(self, query, key):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        return scores

    def forward(self, query, key):
        "Compute 'Scaled Dot Product Attention'"
        query, key = [
            l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))
        ]
        p_attn = self.attention(query, key)
        return p_attn


class enc_mtan(nn.Module):
    def __init__(self, query, embed_time=16, num_heads=1, device="cuda"):
        super(enc_mtan, self).__init__()
        self.embed_time = embed_time
        self.device = device
        self.query = query
        self.att = multiTimeAttention(embed_time, num_heads)
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        # tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, time_steps):
        if len(time_steps.shape) == 4:
            time_steps = time_steps[:, :, 0, :]
        key = self.learn_time_embedding(time_steps).to(self.device)
        query = self.learn_time_embedding(self.query.unsqueeze(0).unsqueeze(-1)).to(
            self.device
        )
        weights = self.att(query, key)
        return weights.permute(0, 3, 1, 2)


class Encoder(nn.Module):
    def __init__(
        self,
        channels,
        hidden_dim,
        hidden_dim_dec,
        kernel_hidden_dim,
        device,
        dropout=0.0,
        mixer_blocks=2,
        add_value_to_weight=True,
        apply_initial_ck=True,
        learhidden_dim=True,
        learn_channel_bias=True,
        share_weights=True,
        fc_mixer=False,
    ):
        super().__init__()
        if share_weights:
            kernel_output_dim = hidden_dim
        else:
            kernel_output_dim = hidden_dim * channels
        self.share_weights = share_weights
        if learhidden_dim:
            self.kernel = nn.Sequential(
                nn.Linear(1, kernel_hidden_dim),
                nn.ReLU(),
                nn.Linear(kernel_hidden_dim, kernel_output_dim),
            )
        else:
            query = torch.arange(0, 0.5, (0.5 + 1e-4) / hidden_dim)
            self.kernel = enc_mtan(
                query=query,
                embed_time=hidden_dim,
                device=device,
            )

        if apply_initial_ck:
            self.init_kernel = nn.Sequential(
                nn.Linear(1, kernel_hidden_dim),
                nn.ReLU(),
                nn.Linear(kernel_hidden_dim, kernel_output_dim),
            )
        self.apply_initial_ck = apply_initial_ck
        if learn_channel_bias:
            self.channel_bias = nn.Parameter(torch.randn(channels, hidden_dim))
        self.channels = channels
        self.out_layer = nn.Sequential(
            nn.RMSNorm([channels, hidden_dim]),
            nn.Linear(hidden_dim, hidden_dim_dec),
        )
        if not fc_mixer:
            if mixer_blocks > 0:
                self.init_channel_mixer = nn.Sequential(
                    nn.RMSNorm([hidden_dim, channels]),
                    nn.Linear(channels, channels),
                    nn.ReLU(),
                )
            self.mixer_blocks = nn.ModuleList(
                [
                    MixerBlock(
                        channels=channels, hidden_dim=hidden_dim, dropout=dropout
                    )
                    for i in range(mixer_blocks - 1)
                ]
            )
        else:
            self.mixer_blocks = nn.ModuleList(
                [
                    FC_mixer(channels=channels, hidden_dim=hidden_dim, dropout=dropout)
                    for i in range(mixer_blocks)
                ]
            )

        self.use_init_mixer = (mixer_blocks > 0) and (not fc_mixer)
        self.add_value_to_weight = add_value_to_weight
        self.learn_channel_bias = learn_channel_bias

    def forward(self, T, X_enc, M):
        # BS x T x C x KD
        init_kernel = self.init_kernel(T.unsqueeze(-1))
        if not self.share_weights:
            init_kernel = init_kernel.unsqueeze(-1).reshape_as(X_enc)
        Z_enc = X_enc * init_kernel
        # BS x T x C x KD
        weights = self.kernel(T.unsqueeze(-1))
        if not self.share_weights:
            weights = weights.unsqueeze(-1).reshape_as(X_enc)
        if self.add_value_to_weight:
            weights = weights + X_enc
        weights = weights - (~M.unsqueeze(-1) * 10e9)
        weights = weights.softmax(dim=1)
        channel_mask = M.sum(1) > 0
        # z: BS x C x KD
        aggregated_channels = ((Z_enc * weights).sum(1)) * channel_mask.unsqueeze(-1)
        z = aggregated_channels
        if self.learn_channel_bias:
            z += self.channel_bias.unsqueeze(0)
        for m in self.mixer_blocks:
            z = z + m(z)
        # BS x C x KD
        if self.use_init_mixer:
            z = z + self.init_channel_mixer(z.transpose(1, 2)).transpose(1, 2)
        z = self.out_layer(z)
        return z


class IMTSMixer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_dim,
        hidden_dim_dec,
        kernel_hidden_dim,
        mixer_blocks,
        device,
        value_embed=True,
        add_value_to_weight=True,
        apply_initial_ck=True,
        learn_channel_bias=True,
        mTAN_kernel=False,
        share_weights=True,
        fc_mixer=False,
    ) -> None:
        super().__init__()

        if mTAN_kernel:
            query = torch.arange(0.5, 1, (0.5 + 1e-4) / hidden_dim_dec)
            self.out_kernel = nn.Sequential(
                enc_mtan(
                    query=query,
                    embed_time=hidden_dim,
                    device=device,
                ),
                nn.Softmax(dim=1),
            )
        else:
            self.out_kernel = nn.Sequential(
                nn.Linear(1, kernel_hidden_dim),
                nn.ReLU(),
                nn.Linear(kernel_hidden_dim, hidden_dim_dec * channels),
            )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim_dec, 1),
        )
        if value_embed:
            self.obs_encoder = nn.Linear(1, hidden_dim)
        learhidden_dim = not mTAN_kernel
        self.value_embed = value_embed
        self.tcnn_encoder = Encoder(
            channels=channels,
            hidden_dim=hidden_dim,
            hidden_dim_dec=hidden_dim_dec,
            kernel_hidden_dim=kernel_hidden_dim,
            mixer_blocks=mixer_blocks,
            device=device,
            add_value_to_weight=add_value_to_weight,
            apply_initial_ck=apply_initial_ck,
            learhidden_dim=learhidden_dim,
            learn_channel_bias=learn_channel_bias,
            share_weights=share_weights,
            fc_mixer=fc_mixer,
        )
        self.hidden_dim = hidden_dim
        self.device = device
        self.learhidden_dim = learhidden_dim
        self.share_weights = share_weights

    def forward(self, T, X, M, YT, MY):
        # X_enc: BS x T xC x KD
        if T.dim() == 2 and self.share_weights:
            T = T.unsqueeze(-1).expand_as(X)
        if self.value_embed:
            X_enc = self.obs_encoder(X.unsqueeze(-1))
        else:
            X_enc = X.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
        z = self.tcnn_encoder(T=T, X_enc=X_enc, M=M)
        # yhat: BS x YT x C x KD
        yhat = z.unsqueeze(1).expand(-1, YT.shape[1], -1, -1)
        if self.learhidden_dim:
            query = self.out_kernel(YT.unsqueeze(-1)).unsqueeze(-1).reshape_as(yhat)
        else:
            query = self.out_kernel(YT.unsqueeze(-1))
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
