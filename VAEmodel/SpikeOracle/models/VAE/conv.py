import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from SpikeOracle.models.VAE.fc import FcNetwork

class ConvNetwork(torch.nn.Module):
    def __init__(self,
                 blocks: int,
                 input_dim: (int, int),
                 image_scaling_factor: float,
                 dropout: float
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.input_seq_len = input_dim[1]
        self.dropout = dropout

        channel_scaling_factor = 1. / image_scaling_factor

        dim_ = input_dim
        self.network = nn.Sequential()
        for block in range(blocks - 1):
            block_, dim_ = self.get_conv_block(dim_, image_scaling_factor, dropout)
            self.network.add_module(f"block_{block}", block_)

        block_, dim_ = self.get_conv_block(dim_, image_scaling_factor, 0)
        self.network.add_module(f"block_{block + 1}", block_)

        self.output_dim = dim_


    def forward(self, x):
        return self.network(x)

    def get_conv_block(self, dim, image_scaling_factor, dropout):
        channel_scaling_factor = 1./image_scaling_factor
        out_dim = [int(dim[0]/image_scaling_factor), dim[1]]

        block = nn.Sequential()
        if image_scaling_factor > 1.:
            block.add_module(f"Upsample", nn.Upsample(scale_factor=image_scaling_factor, mode='linear'))
            if out_dim[1]:
                out_dim[1] *= image_scaling_factor

        block.add_module(f"Conv", nn.Conv1d(dim[0], out_dim[0], kernel_size=3, padding=1))
        block.add_module(f"BN", nn.BatchNorm1d(out_dim[0]))
        block.add_module(f"Activation", nn.LeakyReLU(0.1))

        if dropout > 0:
            block.add_module("Dropout", nn.Dropout(dropout))

        if channel_scaling_factor > 1.:
            block.add_module(f"Pooling", nn.MaxPool1d(int(channel_scaling_factor), stride=int(channel_scaling_factor)))
            if out_dim[1]:
                out_dim[1] = int(out_dim[1] * image_scaling_factor)

        return block, out_dim


class ConvVAE(pl.LightningModule):
    """Standard VAE with Gaussian Prior and approx posterior.
    """

    def __init__(
            self,
            conv_blocks: int,
            conv_input_dim: (int, int),
            conv_image_scaling_factor: float,
            fc_blocks: int,
            fc_hidden_dim: int,
            fc_hidden_dim_scaling_factor: (float, float),
            latent_dim: int,
            conditional: int,
            dropout: float,
            kl_target: float,
            lr: float,
            batch_size: int,
            weight_decay: float
    ):
        super().__init__()

        self.save_hyperparameters()

        self.conv_blocks = conv_blocks
        self.conv_input_dim = conv_input_dim
        self.conv_image_scaling_factor = conv_image_scaling_factor
        self.fc_blocks = fc_blocks
        self.fc_hidden_dim = fc_hidden_dim
        self.fc_hidden_dim_scaling_factor = fc_hidden_dim_scaling_factor
        self.latent_dim = latent_dim
        self.conditional = conditional
        self.dropout = dropout
        self.kl_target = kl_target
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay


        self.beta = 0
        self.P = 0
        self.I = 0

        self.ds = None
        self.dl = None


        self.conv_encoder = ConvNetwork(
            blocks=conv_blocks,
            input_dim=conv_input_dim,
            image_scaling_factor=conv_image_scaling_factor,
            dropout=dropout
        )

        enc_fc_input_dim = self.conv_encoder.output_dim[0]*self.conv_encoder.output_dim[1] + conditional
        enc_output_dim = int(fc_hidden_dim * (fc_hidden_dim_scaling_factor[0]**(fc_blocks - 1)))
        self.fc_encoder = FcNetwork(
            blocks=fc_blocks,
            input_dim=enc_fc_input_dim,
            hidden_dim=fc_hidden_dim,
            hidden_dim_scaling_factor=fc_hidden_dim_scaling_factor[0],
            output_dim=enc_output_dim,
            dropout=dropout
        )

        self.fc_mu = nn.Linear(enc_output_dim, latent_dim)
        self.fc_var = nn.Linear(enc_output_dim, latent_dim)

        dec_fc_input_dim = self.latent_dim + conditional
        dec_fc_output_dim = self.conv_encoder.output_dim[0] * self.conv_encoder.output_dim[1]
        self.fc_decoder = FcNetwork(
            blocks=fc_blocks,
            input_dim=dec_fc_input_dim,
            hidden_dim=int(enc_output_dim * fc_hidden_dim_scaling_factor[1]),
            hidden_dim_scaling_factor=fc_hidden_dim_scaling_factor[1],
            output_dim=dec_fc_output_dim,
            dropout=dropout
        )

        self.conv_decoder_input_dim = self.conv_encoder.output_dim
        self.conv_decoder = ConvNetwork(
            blocks=conv_blocks,
            input_dim=self.conv_decoder_input_dim,
            image_scaling_factor=int(1./conv_image_scaling_factor),
            dropout=dropout
        )

        self.z = None

    def forward(self, x, y, sample=True):
        self.z, x_hat_logit, _, _ = self._run_step(x, y, sample)
        print(x_hat_logit.shape)
        #x_hat_logit = x_hat_logit.reshape(-1, self.conv_input_dim[0], self.conv_input_dim[1])
        x_hat_logit = x_hat_logit.permute(0, 2, 1)
        x_hat = torch.nn.functional.softmax(x_hat_logit, -1)
        return x_hat

    def _run_step(self, x, y, sample=True):
        x = self.conv_encoder(x)

        x = torch.flatten(x, start_dim=-2)
        x = x if self.conditional == 0 else torch.cat([x, y], dim=1)
        x = self.fc_encoder(x)
        x = F.relu(x)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        if sample:
            p, q, z = self.sample(mu, log_var)
        else:
            p, q, z = None, None, mu

        d = z if self.conditional == 0 else torch.cat([z, y], dim=1)

        d = self.fc_decoder(d)
        d = d.reshape(-1, self.conv_decoder_input_dim[0], self.conv_decoder_input_dim[1])
        d = self.conv_decoder(d)

        return z, d, p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch

        z, x_hat_logit, p, q = self._run_step(x, y)

        # recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        recon_loss = F.cross_entropy(
            x_hat_logit.view(-1, self.conv_input_dim[-1]),
            torch.max(x, dim=-1).indices.contiguous().view(-1))

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()

        kl_coeff = self.calc_beta(float(kl.detach()), self.kl_target, 1e-3, 5e-4, 1e-4, 1)

        loss = kl * kl_coeff + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "kl_coeff": kl_coeff,
            "loss": loss,
        }
        return loss, logs

    def calc_beta(self, actual_kl, target_kl, Kp, Ki, beta_min, beta_max):
        error = target_kl - actual_kl
        self.P = Kp / (1 + np.exp(error))

        if beta_min < self.beta and self.beta < beta_max:
            self.I = self.I - Ki * error

        self.beta = min(max(self.P + self.I + beta_min, beta_min), beta_max)
        return self.beta

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        self.dl = DataLoader(self.ds,
                             shuffle=True,
                             batch_size=self.batch_size)
        return self.dl