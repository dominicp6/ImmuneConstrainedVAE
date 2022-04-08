from collections import defaultdict
from tqdm import tqdm
import torch

from SpikeOracle import PHASE_TRAIN, PHASE_VALID, PHASE_TEST


def get_latent_from_seq_FcVAE(VAE, seqs):
    mus = []
    log_vars = []
    latents = []

    Mu_Xs, Mu_Ys, Cats = [], [], []
    Latent_Xs, Latent_Ys = [], []

    VAE.eval()
    ds = VAE.ds[PHASE_TRAIN]

    for seq in seqs:
        x = ds.tok.tokenize(seq).to(VAE.device).unsqueeze(dim=0)
        x = torch.flatten(x, start_dim=-2)
        cat = ds.seq_immuno_cat[seq]
        x = x if VAE.conditional == 0 \
            else torch.cat([x,
                            torch.tensor(ds.seq_immuno_cat_tokens[cat]).unsqueeze(dim=0).to(VAE.device)
                            ], dim=1)

        encoded = VAE.encoder(x)
        mu = VAE.fc_mu(encoded)
        mus.append(mu.detach())
        log_var = VAE.fc_var(encoded)
        log_vars.append(log_var.detach())
        p, q, z = VAE.sample(mu, log_var)
        latents.append(z.detach())

        Mu_Xs.append(float(mu[0][0].detach()))
        Mu_Ys.append(float(mu[0][1].detach()))
        Cats.append(cat)

        Latent_Xs.append(float(z[0][0].detach()))
        Latent_Ys.append(float(z[0][1].detach()))

    return Mu_Xs, Mu_Ys, Latent_Xs, Latent_Ys


def get_seq_from_latent_FcVAE(VAE, latents, condition=None):
    VAE.eval()
    seqs = defaultdict(lambda: 0)
    for z in tqdm(latents):
        if condition is not None:
            h = torch.tensor([[0, 0, 0]])
            h[0][condition] = 1
            z = torch.cat([z.unsqueeze(dim=0), h.to(VAE.device)], dim=1)
        else:
            z = z.unsqueeze(dim=0)

        seq = VAE.decoder(z)
        seq = VAE.ds[PHASE_TRAIN].tok.decode(seq.reshape(1, VAE.sequence_len, -1))[0]
        seqs[seq] += 1

    return seqs
