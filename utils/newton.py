import torch
import torch.nn.functional as F

from utils.data import preprocess


def with_bias(features):
    return torch.cat([features, torch.ones(len(features), 1, device=features.device, dtype=features.dtype)], 1)


def ce_gradients(theta, phi, y):
    p = torch.softmax(phi @ theta.T, 1)
    residual = p - F.one_hot(y, theta.shape[0]).to(p.dtype)
    return (residual[:, :, None] * phi[:, None, :]).reshape(len(y), -1)


def ce_hessian(theta, phi, weights, chunk=2048):
    c, d = theta.shape
    h = torch.zeros(c, d, c, d, dtype=theta.dtype, device=theta.device)
    for i in range(0, len(phi), chunk):
        x, w = phi[i:i + chunk], weights[i:i + chunk]
        p = torch.softmax(x @ theta.T, 1)
        a = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
        h += torch.einsum("nij,nk,nl->ikjl", a * w[:, None, None], x, x)
    return h.reshape(c * d, c * d) / len(phi)


@torch.no_grad()
def body_features(model, benchmark, x, batch_size=500):
    model.eval()
    return torch.cat([model.features(preprocess(benchmark, x[i:i + batch_size], train=False)).double()
                      for i in range(0, len(x), batch_size)])


def counterfactual_bias(theta, phi_c, phi_cbar):
    p_c, p_cbar = torch.softmax(phi_c @ theta.T, 1), torch.softmax(phi_cbar @ theta.T, 1)
    pred = p_c.argmax(1, keepdim=True)
    return (p_c.gather(1, pred) - p_cbar.gather(1, pred)).abs().mean().item()
