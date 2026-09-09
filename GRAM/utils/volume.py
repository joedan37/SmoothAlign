import torch
import numpy as np

# * * * *  * * * *  * * * *   *       *   
# *        *     *  *     *   * *   * *   
# *   * *  * * *    * * * *   *   *   *   
# *     *  *     *  *     *   *       *   
# * * * *  *     *  *     *   *       *   

# THIS IS THE CORE PY CODE OF GRAM FRAMEWORK


def _stable_gram_volume(G, output_dtype):
    """Compute small Gram determinants off CUDA to avoid MAGMA batched-LU crashes."""
    gram_det = torch.linalg.det(G.float().cpu()).to(G.device)
    gram_det = torch.clamp(gram_det, min=0.0)
    return torch.sqrt(gram_det).to(dtype=output_dtype)


def _pool_prior_feature(feature):
    if feature.dim() > 2:
        feature = feature.reshape(feature.shape[0], -1, feature.shape[-1]).mean(dim=1)
    return feature.float()


def compute_gaussian_prior(
    modality_features,
    tau=1.0,
    eps=1e-6,
    distance_type="symmetric_kl",
):
    """
    Build a batch-level prior matrix from the Gaussian fitted over each sample's modalities.

    Args:
    - modality_features: list of tensors. Each tensor starts with batch dimension N and
      can be either [N, D] or token/frame features [N, ..., D].
    - tau: temperature for converting distances to affinities.
    - eps: numerical floor for variance and division.
    - distance_type: one of symmetric_kl, mean_cosine, wasserstein2, or csd.

    Returns:
    - torch.Tensor: [N, N] prior similarity matrix with diagonal fixed to 1.
    """
    valid_features = [f for f in modality_features if f is not None]
    if len(valid_features) < 2:
        raise ValueError("compute_gaussian_prior needs at least two modalities.")

    pooled_features = [_pool_prior_feature(f) for f in valid_features]
    pooled_features = [torch.nn.functional.normalize(f, dim=-1) for f in pooled_features]
    stacked_features = torch.stack(pooled_features, dim=0)  # M, N, D

    mu = stacked_features.mean(dim=0)
    var = stacked_features.var(dim=0, unbiased=False).clamp_min(eps)

    distance_type = str(distance_type).lower()
    mu_i = mu.unsqueeze(1)
    mu_j = mu.unsqueeze(0)
    var_i = var.unsqueeze(1)
    var_j = var.unsqueeze(0)

    if distance_type == "symmetric_kl":
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
        distance = 0.5 * (kl_ij + kl_ji)
    elif distance_type == "mean_cosine":
        normalized_mu = torch.nn.functional.normalize(mu, dim=-1)
        distance = (1.0 - normalized_mu @ normalized_mu.T).clamp(min=0.0, max=2.0)
    elif distance_type == "wasserstein2":
        std_i = torch.sqrt(var_i)
        std_j = torch.sqrt(var_j)
        distance = ((mu_i - mu_j).pow(2) + (std_i - std_j).pow(2)).sum(dim=-1)
    elif distance_type == "csd":
        distance = ((mu_i - mu_j).pow(2) + var_i + var_j).sum(dim=-1)
    else:
        raise ValueError(
            "Unsupported Gaussian prior distance_type: "
            f"{distance_type}. Expected symmetric_kl, mean_cosine, wasserstein2, or csd."
        )

    prior = torch.exp(-distance / max(float(tau), eps))
    prior = prior.clamp(min=0.0, max=1.0)
    prior.fill_diagonal_(1.0)
    return prior.to(dtype=valid_features[0].dtype)



def volume_computation3(language, video, audio):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 3x3
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    


    # Explicit 3x3 Gram determinant avoids CUDA MAGMA batched LU instability.
    gram_det = (
        ll.float() * (vv.float() * aa.float() - va.float().pow(2))
        - lv.float() * (lv.float() * aa.float() - va.float() * la.float())
        + la.float() * (lv.float() * va.float() - vv.float() * la.float())
    )
    gram_det = torch.clamp(gram_det, min=0.0)

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(gram_det)
    #print(res.shape)
    return res


def volume_computation4(language, video, audio, subtitles):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 4x4
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio, and subtitles (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T
    ls = language@subtitles.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    
    ss = torch.einsum('bi,bi->b', subtitles, subtitles).unsqueeze(0).expand(batch_size1, -1)
    vs = torch.einsum('bi,bi->b', video, subtitles).unsqueeze(0).expand(batch_size1, -1)
    sa = torch.einsum('bi,bi->b', audio, subtitles).unsqueeze(0).expand(batch_size1, -1)

    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 4, 4])
    G = torch.stack([
        torch.stack([ll, lv, la, ls], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv, va, vs], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, aa, sa], dim=-1),  # Third row of the Gram matrix
        torch.stack([ls, vs, sa, ss], dim=-1)   # Fourth row of the Gram matrix
    ], dim=-2)

    res = _stable_gram_volume(G, language.dtype)
    #print(res.shape)
    return res


def volume_computation5(language, video, audio, subtitles, depth):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 5x5
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    - depth (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing depth features.    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio, and subtitles (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T
    ls = language@subtitles.T
    ld = language@depth.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    
    
    ss = torch.einsum('bi,bi->b', subtitles, subtitles).unsqueeze(0).expand(batch_size1, -1)
    vs = torch.einsum('bi,bi->b', video, subtitles).unsqueeze(0).expand(batch_size1, -1)
    sa = torch.einsum('bi,bi->b', audio, subtitles).unsqueeze(0).expand(batch_size1, -1)

    dd = torch.einsum('bi,bi->b', depth, depth).unsqueeze(0).expand(batch_size1, -1)
    dv = torch.einsum('bi,bi->b', depth, video).unsqueeze(0).expand(batch_size1, -1)
    da = torch.einsum('bi,bi->b', depth, audio).unsqueeze(0).expand(batch_size1, -1) 
    ds = torch.einsum('bi,bi->b', depth, subtitles).unsqueeze(0).expand(batch_size1, -1)


    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 5, 5])
    G = torch.stack([
        torch.stack([ll, lv, la, ls, ld], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv, va, vs, dv], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, aa, sa, da], dim=-1),  # Third row of the Gram matrix
        torch.stack([ls, vs, sa, ss, ds], dim=-1),   # Fourth row of the Gram matrix
        torch.stack([ld, dv, da, ds, dd], dim=-1)
    ], dim=-2)

    res = _stable_gram_volume(G, language.dtype)
    #print(res.shape)
    return res


def volume_computation(language, *inputs):
    """
    General function to compute volume for contrastive learning loss functions.
    Compute the volume metric for each vector in language batch and all the other modalities listed in *inputs.

    Args:
    - language (torch.Tensor): Tensor of shape (batch_size1, dim)
    - *inputs (torch.Tensor): Variable number of tensors of shape (batch_size2, dim)

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """
    batch_size1 = language.shape[0]
    batch_size2 = inputs[0].shape[0]

    # Compute pairwise dot products for language with itself
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with each input
    l_inputs = [language @ input.T for input in inputs]

    # Compute pairwise dot products for each input with themselves and with each other
    input_dot_products = []
    for i, input1 in enumerate(inputs):
        row = []
        for j, input2 in enumerate(inputs):
            dot_product = torch.einsum('bi,bi->b', input1, input2).unsqueeze(0).expand(batch_size1, -1)
            row.append(dot_product)
        input_dot_products.append(row)

    # Stack the results to form the Gram matrix for each pair
    G = torch.stack([
        torch.stack([ll] + l_inputs, dim=-1),
        *[torch.stack([l_inputs[i]] + input_dot_products[i], dim=-1) for i in range(len(inputs))]
    ], dim=-2)

    return _stable_gram_volume(G, language.dtype)
