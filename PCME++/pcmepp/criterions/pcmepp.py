""" Improved probabilistic embedding loss for cross-modal retrieval

PCME++
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClosedFormSampledDistanceLoss(nn.Module):
    def __init__(
            self,
            init_shift=5,
            init_negative_scale=5,
            vib_beta=0,
            smoothness_alpha=0,
            prob_distance='csd',
            smooth_prior=False,
            prior_tau=1e4,
            prior_eps=1e-6,
            confidence_enabled=True,
            confidence_temp=0.07,
            confidence_power=0.5,
            confidence_floor=0.0,
            min_negative_weight=1e-6,
            **kwargs):
        super().__init__()

        shift = init_shift * torch.ones(1)
        negative_scale = init_negative_scale * torch.ones(1)

        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)

        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

        self.vib_beta = vib_beta
        self.smoothness_alpha = smoothness_alpha
        self.smooth_prior = bool(smooth_prior)
        self.prior_tau = float(prior_tau)
        self.prior_eps = float(prior_eps)
        self.confidence_enabled = bool(confidence_enabled)
        self.confidence_temp = float(confidence_temp)
        self.confidence_power = float(confidence_power)
        self.confidence_floor = float(confidence_floor)
        self.min_negative_weight = float(min_negative_weight)

        # XXX Do not specify prob_distance unless for the prob dist ablation study
        self.prob_distance = prob_distance

        self.bceloss = nn.BCEWithLogitsLoss()

        if self.prob_distance not in {'csd', 'wdist'}:
            raise ValueError(f'Invalid prob_distance. Expected ("csd", "wdist"), but {prob_distance=}')

    def max_violation_on(self):
        warnings.warn(
            'PCME loss does not support max violation. Nothing happens')
        return

    def max_violation_off(self):
        warnings.warn(
            'PCME loss does not support max violation. Nothing happens')
        return

    def kl_divergence(self, mu, logsigma):
        kl_loss = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).mean()
        if kl_loss > 10000:
            # XXX prevent loss exploration
            warnings.warn(f'Detected a VIB loss explosion ({kl_loss=} > 10000). Ignore the VIB loss for stability.')
            return 0
        return kl_loss

    def _recompute_matched(self, matched, logits, smoothness=0):
        """ Recompute the `matched` matrix if the smoothness value is given.
        """
        if not smoothness:
            return matched, None
        else:
            logits = logits.view(matched.size())
            # XXX Warning: all negative pairs will return weird results
            gt_labels, gt_indices = torch.max(matched, dim=1)
            gt_vals = logits[:, gt_indices].diag()
            pseudo_gt_indices = (logits >= gt_vals.unsqueeze(1))
            new_matched = (gt_labels.unsqueeze(1) * (pseudo_gt_indices))
            _matched = matched.clone()
            _matched[pseudo_gt_indices] = new_matched[pseudo_gt_indices]

            return _matched, torch.sum(pseudo_gt_indices).item() - len(gt_indices)

    def _compute_prob_matching_loss(
            self, logits, matched, smoothness=0, prior_matrix=None):
        matched, n_pseudo_gts = self._recompute_matched(matched, logits, smoothness)
        if prior_matrix is None:
            loss = self.bceloss(logits, matched)
        else:
            if prior_matrix.shape != logits.shape:
                raise ValueError(
                    f'Prior/logit shape mismatch: {prior_matrix.shape} vs '
                    f'{logits.shape}')
            prior_matrix = prior_matrix.to(
                device=logits.device, dtype=logits.dtype)
            negative_weights = (1.0 - prior_matrix).clamp(
                min=max(self.min_negative_weight, 1e-12), max=1.0)
            entry_loss = F.binary_cross_entropy_with_logits(
                logits, matched, reduction='none')
            weights = torch.where(
                matched <= 0.5, negative_weights, torch.ones_like(logits))
            loss = (entry_loss * weights).mean()

        loss_dict = {
            'loss': loss,
            'n_pseudo_gts': n_pseudo_gts,
        }
        if prior_matrix is not None:
            negative_mask = matched <= 0.5
            negative_prior = prior_matrix.masked_select(negative_mask)
            negative_weights = (1.0 - negative_prior).clamp(
                min=max(self.min_negative_weight, 1e-12), max=1.0)
            loss_dict['smooth/prior_mean'] = negative_prior.mean()
            loss_dict['smooth/negative_weight_mean'] = negative_weights.mean()
        return loss_dict

    def _compute_closed_form_loss(
            self, input1, input2, matched, smoothness=0, prior_matrix=None):
        """ Closed-form probabilistic matching loss -- See Eq (1) and (2) in the paper.
        """
        mu_pdist = ((input1['mean'].unsqueeze(1) - input2['mean'].unsqueeze(0)) ** 2).sum(-1)
        sigma_pdist = ((torch.exp(input1['std']).unsqueeze(1) + torch.exp(input2['std']).unsqueeze(0))).sum(-1)
        logits = mu_pdist + sigma_pdist
        logits = -self.negative_scale * logits + self.shift
        loss_dict = self._compute_prob_matching_loss(
            logits, matched, smoothness=smoothness,
            prior_matrix=prior_matrix)
        loss_dict['loss/mu_pdist'] = mu_pdist.mean()
        loss_dict['loss/sigma_pdist'] = sigma_pdist.mean()
        return loss_dict

    def _compute_wd_loss(
            self, input1, input2, matched, smoothness=0, prior_matrix=None):
        """ Wasserstien loss (only used for the ablation study)
        """
        mu_pdist = ((input1['mean'].unsqueeze(1) - input2['mean'].unsqueeze(0)) ** 2).sum(-1).view(-1)
        sigma_pdist = ((torch.exp(input1['std'] / 2).unsqueeze(1) - torch.exp(input2['std'] / 2).unsqueeze(0)) ** 2).sum(-1).view(-1)

        logits = mu_pdist + sigma_pdist
        logits = logits.reshape(len(input1['mean']), len(input2['mean']))
        logits = -self.negative_scale * logits + self.shift
        loss_dict = self._compute_prob_matching_loss(
            logits, matched, smoothness=smoothness,
            prior_matrix=prior_matrix)
        loss_dict['loss/mu_pdist'] = mu_pdist.mean()
        loss_dict['loss/sigma_pdist'] = sigma_pdist.mean()
        return loss_dict

    @torch.no_grad()
    def build_prior(self, image_features, caption_features):
        """Build a detached Gaussian semantic prior from paired features."""
        if image_features.ndim != 2 or caption_features.ndim != 2:
            raise ValueError('Smooth prior expects [batch, dimension] features')
        if image_features.shape != caption_features.shape:
            raise ValueError(
                'Smooth prior needs paired image/text features with equal shapes')

        eps = max(self.prior_eps, 1e-12)
        image_features = F.normalize(image_features.detach().float(), dim=-1)
        caption_features = F.normalize(caption_features.detach().float(), dim=-1)
        stacked_features = torch.stack(
            [image_features, caption_features], dim=0)
        mu = stacked_features.mean(dim=0)
        var = stacked_features.var(
            dim=0, unbiased=False).clamp_min(eps)

        mu_i, mu_j = mu.unsqueeze(1), mu.unsqueeze(0)
        var_i, var_j = var.unsqueeze(1), var.unsqueeze(0)
        kl_ij = 0.5 * (
            var_i / var_j
            + (mu_j - mu_i).pow(2) / var_j
            - 1.0
            + torch.log(var_j / var_i)
        ).sum(dim=-1)
        kl_ji = 0.5 * (
            var_j / var_i
            + (mu_i - mu_j).pow(2) / var_i
            - 1.0
            + torch.log(var_i / var_j)
        ).sum(dim=-1)
        prior = torch.exp(-0.5 * (kl_ij + kl_ji) / max(self.prior_tau, eps))
        prior = prior.clamp(min=0.0, max=1.0)

        if self.confidence_enabled:
            confidence_temp = max(self.confidence_temp, eps)
            logits = image_features @ caption_features.T / confidence_temp
            row_prob = F.softmax(logits, dim=1).diag()
            col_prob = F.softmax(logits, dim=0).diag()
            confidence = torch.sqrt((row_prob * col_prob).clamp_min(0.0))
            batch_size = image_features.size(0)
            if batch_size > 1:
                random_prob = 1.0 / batch_size
                confidence = (
                    (confidence - random_prob) / (1.0 - random_prob)
                ).clamp(0.0, 1.0)
            confidence = confidence.pow(max(self.confidence_power, eps))
            floor = min(max(self.confidence_floor, 0.0), 1.0)
            if floor > 0:
                confidence = floor + (1.0 - floor) * confidence
            prior = prior * torch.sqrt(
                confidence.unsqueeze(1) * confidence.unsqueeze(0))

        prior.fill_diagonal_(1.0)
        return prior.to(dtype=image_features.dtype)

    def forward(self, img_emb, cap_emb, matched=None, prior_matrix=None):
        if self.prob_distance == 'wdist':
            loss_fn = self._compute_wd_loss
        else:
            loss_fn = self._compute_closed_form_loss
        vib_loss = 0

        if self.vib_beta != 0:
            vib_loss =\
                self.kl_divergence(img_emb['mean'], img_emb['std']) + \
                self.kl_divergence(cap_emb['mean'], cap_emb['std'])

        if matched is None:
            matched = torch.eye(len(img_emb['mean'])).to(img_emb['mean'].device)

        if self.smooth_prior and prior_matrix is None:
            if 'prior' not in img_emb or 'prior' not in cap_emb:
                raise ValueError(
                    'PCME++ + Smooth requires detached prior features')
            prior_matrix = self.build_prior(img_emb['prior'], cap_emb['prior'])

        loss = loss_fn(
            img_emb, cap_emb, matched=matched,
            prior_matrix=prior_matrix if self.smooth_prior else None)
        # NOTE: Efficient implementation for
        # when i2t loss and t2i loss are the same (https://github.com/naver-ai/pcme/issues/3)
        loss = 2 * loss['loss'] + self.vib_beta * vib_loss

        loss_dict = {
            'loss/loss': loss,
            'criterion/shift': self.shift,
            'criterion/negative_scale': self.negative_scale,
        }

        if self.vib_beta != 0:
            loss_dict['loss/vib_loss'] = vib_loss

        if self.smoothness_alpha:
            smooth_i2t_loss = loss_fn(img_emb, cap_emb, matched=matched, smoothness=self.smoothness_alpha)
            smooth_t2i_loss = loss_fn(cap_emb, img_emb, matched=matched.T, smoothness=self.smoothness_alpha)
            loss = loss + self.smoothness_alpha * (smooth_i2t_loss['loss'] + smooth_t2i_loss['loss'])
            loss_dict['loss/loss'] = loss
            loss_dict['loss/n_pseudo_gts'] = smooth_i2t_loss['n_pseudo_gts'] + smooth_t2i_loss['n_pseudo_gts']

        return loss, loss_dict
