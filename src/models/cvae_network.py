"""Forecasting numerical definitions adapted from the original stage 1 scripts."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(a, b, width=128):
    return nn.Sequential(
        nn.Linear(a, width),
        nn.SiLU(),
        nn.Linear(width, width),
        nn.SiLU(),
        nn.Linear(width, b),
    )


class CVAE(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.prior = mlp(c, 16)
        self.posterior = mlp(d + c, 16)
        self.decoder = mlp(8 + c, d)
        self.log_noise = nn.Parameter(torch.full((d,), -1.0))

    def loss(self, z, c):
        qm, ql = self.posterior(torch.cat([z, c], -1)).chunk(2, -1)
        pm, pl = self.prior(c).chunk(2, -1)
        ql = ql.clamp(-6, 3)
        pl = pl.clamp(-6, 3)
        eps = torch.randn_like(qm)
        latent = qm + eps * torch.exp(0.5 * ql)
        out = self.decoder(torch.cat([latent, c], -1))
        noise = self.log_noise.clamp(-4, 2)
        recon = ((z - out) ** 2 * torch.exp(-2 * noise) + 2 * noise).mean() / 2
        kl = 0.5 * (pl - ql + (ql.exp() + (qm - pm) ** 2) / pl.exp() - 1).mean()
        return recon + 0.1 * kl, recon, kl

    def sample(self, c, n):
        pm, pl = self.prior(c).chunk(2, -1)
        pm = pm.repeat_interleave(n, 0)
        pl = pl.clamp(-6, 3).repeat_interleave(n, 0)
        cc = c.repeat_interleave(n, 0)
        z = pm + torch.randn_like(pm) * torch.exp(0.5 * pl)
        out = self.decoder(torch.cat([z, cc], -1))
        return out + torch.randn_like(out) * self.log_noise.clamp(-4, 2).exp()

    def reconstruction(self, z, c):
        qm, _ = self.posterior(torch.cat([z, c], -1)).chunk(2, -1)
        return self.decoder(torch.cat([qm, c], -1))
