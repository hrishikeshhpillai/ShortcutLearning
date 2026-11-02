import torch
import torch.nn.functional as F

from poolforge.methods.lff import LfF, generalized_ce


class LogitCorrection(LfF):
    def __init__(self, ctx):
        super().__init__(ctx)
        c = ctx.num_classes
        self.prior = torch.full((c, c), 1.0 / (c * c), device=ctx.device)

    def corrected(self, logits, log_prior):
        return logits + log_prior

    def step(self, x, y, index, step):
        args, c = self.ctx.args, self.ctx.num_classes
        logit_b, logit_d = self.model_b(x), self.model(x)
        self.ema_b.update(F.cross_entropy(logit_b, y, reduction="none").detach(), index)
        self.ema_d.update(F.cross_entropy(logit_d, y, reduction="none").detach(), index)
        attr = logit_b.detach().argmax(1)
        batch_prior = torch.zeros(c, c, device=x.device).index_put_((y, attr), torch.ones_like(y, dtype=torch.float),
                                                                   accumulate=True) / len(y)
        self.prior = args.lc_alpha * self.prior + (1 - args.lc_alpha) * batch_prior
        log_prior = torch.log(self.prior.clamp(min=1e-8)).T
        loss = generalized_ce(logit_b, y).mean() + F.cross_entropy(logit_d + args.lc_eta * log_prior[attr], y)
        if args.lc_mix > 0:
            minority = attr != y
            same = (y[:, None] == y[None, :]) & (attr == y)[None, :]
            rows = (minority[:, None] & same).float()
            has = rows.sum(1) > 0
            if has.any():
                i = has.nonzero().squeeze(1)
                j = torch.multinomial(rows[i], 1).squeeze(1)
                lam = 0.5 + 0.5 * torch.rand(len(i), device=x.device)
                shape = (-1,) + (1,) * (x.dim() - 1)
                x_mix = lam.view(shape) * x[i] + (1 - lam.view(shape)) * x[j]
                prior_mix = lam[:, None] * self.prior.T[attr[i]] + (1 - lam[:, None]) * self.prior.T[attr[j]]
                loss = loss + args.lc_mix * F.cross_entropy(
                    self.model(x_mix) + args.lc_eta * torch.log(prior_mix.clamp(min=1e-8)), y[i])
        self.optimizer_b.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer_b.step()
        self.optimizer.step()
        return logit_d
