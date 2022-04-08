import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from SpikeOracle import PHASE_TRAIN, PHASE_VALID, PHASE_TEST


class FcNetwork(torch.nn.Module):
    def __init__(self,
                 blocks: int,
                 input_dim: int,
                 hidden_dim: int,
                 hidden_dim_scaling_factor: float,
                 output_dim: int,
                 dropout: float
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim != None \
                                     else int(hidden_dim * (hidden_dim_scaling_factor**(blocks - 1)))
        self.dropout = dropout

        input_dim_ = input_dim
        output_dim_ = hidden_dim
        self.network = nn.Sequential()
        for block in range(blocks - 1):
            self.network.add_module(f"block_{block}", self.get_fc_block(input_dim_, output_dim_, dropout))
            input_dim_ = int(output_dim_)
            output_dim_ = int(output_dim_ * hidden_dim_scaling_factor)
        self.network.add_module(f"block_{block + 1}", self.get_fc_block(input_dim_, self.output_dim, 0))

    def forward(self, x):
        return self.network(x)

    def get_fc_block(self, input_dim, output_dim, dropout=0):
        block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.1)
        )
        if dropout > 0:
            block.add_module(str(3), nn.Dropout(dropout))

        return block


class FcVAE(pl.LightningModule):
    """Standard VAE with Gaussian Prior and approx posterior.
    """

    def __init__(
            self,
            aa_dim: int,
            sequence_len: int,
            blocks: int,
            hidden_dim: int,
            hidden_dim_scaling_factor: (float, float),
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

        self.aa_dim = aa_dim
        self.sequence_len = sequence_len
        self.blocks = blocks
        self.hidden_dim = hidden_dim
        self.hidden_dim_scaling_factor = hidden_dim_scaling_factor
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

        self.ds = [None, None, None]
        self.dl = [None, None, None]

        enc_input_dim = self.aa_dim * sequence_len + conditional
        enc_output_dim = int(hidden_dim * (hidden_dim_scaling_factor[0]**(blocks - 1)))
        self.encoder = FcNetwork(
            blocks=blocks,
            input_dim=enc_input_dim,
            hidden_dim=hidden_dim,
            hidden_dim_scaling_factor=hidden_dim_scaling_factor[0],
            output_dim=enc_output_dim,
            dropout=dropout
        )

        self.fc_mu = nn.Linear(enc_output_dim, latent_dim)
        self.fc_var = nn.Linear(enc_output_dim, latent_dim)

        dec_input_dim = self.latent_dim + conditional
        dec_output_dim = self.aa_dim * sequence_len
        self.decoder = FcNetwork(
            blocks=blocks,
            input_dim=dec_input_dim,
            hidden_dim=int(enc_output_dim * hidden_dim_scaling_factor[1]),
            hidden_dim_scaling_factor=hidden_dim_scaling_factor[1],
            output_dim=dec_output_dim,
            dropout=dropout
        )

        self.z = None

    def forward(self, x, y, sample=True):
        self.z, x_hat_logit, _, _ = self._run_step(x, y, sample)
        x_hat_logit = x_hat_logit.reshape(-1, self.sequence_len, self.aa_dim)
        x_hat = torch.nn.functional.softmax(x_hat_logit, -1)
        return x_hat

    def _run_step(self, x, y, sample=True):
        x = torch.flatten(x, start_dim=-2)
        x = x if self.conditional == 0 else torch.cat([x, y], dim=1)

        x = self.encoder(x)
        x = F.relu(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        if sample:
            p, q, z = self.sample(mu, log_var)
        else:
            p, q, z = None, None, mu

        d = z if self.conditional == 0 else torch.cat([z, y], dim=1)
        return z, self.decoder(d), p, q

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
            x_hat_logit.view(-1, self.aa_dim),
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

    def get_dataloader(self, phase):
        shuffle = True if phase == PHASE_TRAIN else False
        self.dl[phase] = DataLoader(self.ds[phase],
                             shuffle=shuffle,
                             batch_size=self.batch_size)
        return self.dl[phase]

    def train_dataloader(self):
        return self.get_dataloader(PHASE_TRAIN)

    def val_dataloader(self):
        return self.get_dataloader(PHASE_VALID)

    def test_dataloader(self):
        return self.get_dataloader(PHASE_TEST)

    def get_latent_from_seq(self, seqs):
        mus = []
        log_vars = []
        latents = []
        cats = []

        self.eval()
        ds = self.ds[PHASE_TRAIN]

        for seq in seqs:
            x = ds.tok.tokenize(seq).to(self.device).unsqueeze(dim=0)
            x = torch.flatten(x, start_dim=-2)
            cat = ds.seq_immuno_cat[seq]
            x = x if self.conditional == 0 \
                else torch.cat([x,
                                torch.tensor(ds.seq_immuno_cat_tokens[cat]).unsqueeze(dim=0).to(self.device)
                                ], dim=1)

            encoded = self.encoder(x)
            mu = self.fc_mu(encoded)
            log_var = self.fc_var(encoded)

            p, q, z = self.sample(mu, log_var)

            mus.append(mu.detach())
            log_vars.append(log_var.detach())
            latents.append(z.detach())
            cats.append(cat)

        return mus, log_vars, latents, cats

    def get_seq_from_latent(self, latents, condition=None):
        self.eval()
        seqs = defaultdict(lambda: 0)
        for z in tqdm(latents):
            if condition is not None:
                h = torch.tensor([[0, 0, 0]])
                h[0][condition] = 1
                z = torch.cat([z.unsqueeze(dim=0), h.to(self.device)], dim=1)
            else:
                z = z.unsqueeze(dim=0)

            seq = self.decoder(z)
            seq = self.ds[PHASE_TRAIN].tok.decode(seq.reshape(1, self.sequence_len, -1))[0]
            seqs[seq] += 1

        return seqs