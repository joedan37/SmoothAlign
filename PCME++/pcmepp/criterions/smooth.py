"""Confidence-gated probabilistic smoothing for contrastive alignment."""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothLoss(nn.Module):
    """Bidirectional contrastive loss with a multimodal Gaussian prior.

    Each matched image--caption pair is treated as two observations of one
    semantic event. Their normalized representations define a diagonal
    Gaussian cloud. Symmetric KL divergence between clouds provides a
    detached semantic affinity, which downweights only negative terms in the
    contrastive denominator. Setting every prior entry to zero recovers the
    standard InfoNCE objective.
    """

    def __init__(
            self,
            init_tau=1,
            prior_tau=1.0,
            prior_eps=1e-6,
            confidence_enabled=True,
            confidence_temp=0.07,
            confidence_power=0.5,
            confidence_floor=0.0,
            min_negative_weight=1e-6,
            **kwargs):
        super().__init__()

        self.register_parameter(
            'tau', nn.Parameter(init_tau * torch.ones(1)))
        self.prior_tau = float(prior_tau)
        self.prior_eps = float(prior_eps)
        self.confidence_enabled = bool(confidence_enabled)
        self.confidence_temp = float(confidence_temp)
        self.confidence_power = float(confidence_power)
        self.confidence_floor = float(confidence_floor)
        self.min_negative_weight = float(min_negative_weight)

    def max_violation_on(self):
        warnings.warn(
            'Smooth loss does not support max violation. Nothing happens')

    def max_violation_off(self):
        warnings.warn(
            'Smooth loss does not support max violation. Nothing happens')

    @torch.no_grad()
    def build_prior(self, image_features, caption_features):
        """Construct the confidence-gated sample affinity matrix."""
        if image_features.ndim != 2 or caption_features.ndim != 2:
            raise ValueError('Smooth prior expects [batch, dimension] features')
        if image_features.shape != caption_features.shape:
            raise ValueError(
                'Smooth prior needs paired image/text features with equal shapes, '
                f'got {image_features.shape} and {caption_features.shape}')

        eps = max(self.prior_eps, 1e-12)
        image_features = F.normalize(image_features.detach().float(), dim=-1)
        caption_features = F.normalize(caption_features.detach().float(), dim=-1)
        stacked_features = torch.stack(
            [image_features, caption_features], dim=0)

        mu = stacked_features.mean(dim=0)
        var = stacked_features.var(
            dim=0, unbiased=False).clamp_min(eps)

        mu_i = mu.unsqueeze(1)
        mu_j = mu.unsqueeze(0)
        var_i = var.unsqueeze(1)
        var_j = var.unsqueeze(0)

        kl_ij = 0.5 * (
            (var_i / var_j)
            + ((mu_j - mu_i).pow(2) / var_j)
            - 1.0
            + torch.log(var_j / var_i)
        ).sum(dim=-1)
        kl_ji = 0.5 * (
            (var_j / var_i)
            + ((mu_i - mu_j).pow(2) / var_i)
            - 1.0
            + torch.log(var_i / var_j)
        ).sum(dim=-1)
        symmetric_kl = 0.5 * (kl_ij + kl_ji)

        prior_tau = max(self.prior_tau, eps)
        prior = torch.exp(-symmetric_kl / prior_tau)
        prior = prior.clamp(min=0.0, max=1.0)

        if self.confidence_enabled:
            confidence_temp = max(self.confidence_temp, eps)
            logits = image_features @ caption_features.T
            logits = logits / confidence_temp
            row_prob = F.softmax(logits, dim=1).diag()
            col_prob = F.softmax(logits, dim=0).diag()
            sample_confidence = torch.sqrt(
                (row_prob * col_prob).clamp_min(0.0))

            batch_size = image_features.size(0)
            if batch_size > 1:
                random_prob = 1.0 / batch_size
                sample_confidence = (
                    (sample_confidence - random_prob)
                    / (1.0 - random_prob)
                ).clamp(0.0, 1.0)

            confidence_power = max(self.confidence_power, eps)
            sample_confidence = sample_confidence.pow(confidence_power)
            confidence_floor = min(max(self.confidence_floor, 0.0), 1.0)
            if confidence_floor > 0:
                sample_confidence = confidence_floor + (
                    1.0 - confidence_floor) * sample_confidence

            pair_confidence = torch.sqrt(
                sample_confidence.unsqueeze(1)
                * sample_confidence.unsqueeze(0)
            )
            prior = prior * pair_confidence

        prior.fill_diagonal_(1.0)
        return prior.to(dtype=image_features.dtype)

    def _soften_logits(self, logits, prior_matrix, targets):
        if prior_matrix.shape != logits.shape:
            raise ValueError(
                f'Prior/logit shape mismatch: {prior_matrix.shape} vs '
                f'{logits.shape}')

        prior_matrix = prior_matrix.to(
            device=logits.device, dtype=logits.dtype)
        negative_weights = (1.0 - prior_matrix).clamp(
            min=max(self.min_negative_weight, 1e-12), max=1.0)
        softened_logits = logits + negative_weights.log()

        positive_logits = logits.gather(1, targets.unsqueeze(1))
        softened_logits = softened_logits.scatter(
            1, targets.unsqueeze(1), positive_logits)
        return softened_logits, negative_weights

    def forward(
            self,
            img_emb,
            cap_emb,
            distributed=False,
            prior_matrix=None,
            **kwargs):
        image_features = img_emb['mean']
        caption_features = cap_emb['mean']
        logits = (image_features @ caption_features.T) * self.tau
        targets = torch.arange(
            logits.size(0), dtype=torch.long, device=logits.device)

        if prior_matrix is None:
            prior_image_features = img_emb.get('prior', image_features)
            prior_caption_features = cap_emb.get('prior', caption_features)
            if prior_image_features.shape != prior_caption_features.shape:
                raise ValueError(
                    'Distributed Smooth loss requires an explicitly aligned '
                    'prior matrix')
            prior_matrix = self.build_prior(
                prior_image_features, prior_caption_features)

        softened_logits, negative_weights = self._soften_logits(
            logits, prior_matrix, targets)
        if distributed:
            loss = F.cross_entropy(softened_logits, targets)
        else:
            softened_logits_t, _ = self._soften_logits(
                logits.T, prior_matrix.T, targets)
            loss = (
                F.cross_entropy(softened_logits, targets)
                + F.cross_entropy(softened_logits_t, targets)
            ) / 2

        positive_mask = torch.zeros_like(
            prior_matrix, dtype=torch.bool)
        positive_mask.scatter_(1, targets.unsqueeze(1), True)
        negative_prior = prior_matrix.masked_select(~positive_mask)
        negative_weight = negative_weights.masked_select(~positive_mask)

        loss_dict = {
            'loss/loss': loss,
            'criterion/tau': self.tau,
            'smooth/prior_mean': negative_prior.mean(),
            'smooth/prior_max': negative_prior.max(),
            'smooth/prior_p90': torch.quantile(negative_prior.float(), 0.90),
            'smooth/prior_p99': torch.quantile(negative_prior.float(), 0.99),
            'smooth/negative_weight_mean': negative_weight.mean(),
            'smooth/softened_fraction': (
                negative_weight < 1.0 - 1e-6).float().mean(),
        }
        return loss, loss_dict
