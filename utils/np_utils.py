import torch
import matplotlib.pyplot as plt


def sample_ctx_tgt(
    img_flat: torch.Tensor,
    img_size: int,
    min_ctx: int = 10,
    max_ctx_frac: float = 0.25,
    tgt_frac: float = 1/3,
):
    """
    Split flattened images into random context/target pixel sets.

    Args:
        img_flat: (B, N, C) tensor, N = img_size**2, values in [-1, 1].
        img_size: Image height/width (H = W).
        min_ctx: Min number of context points.
        max_ctx_frac: Max context points as fraction of all pixels.
        tgt_frac: Target points as fraction of all pixels.

    Returns:
        ctx_x: (B, ctx_size, 2)  normalized coords in [-1, 1]^2
        ctx_y: (B, ctx_size, C)  pixel values
        tgt_x: (B, tgt_size_eff, 2)
        tgt_y: (B, tgt_size_eff, C)
        pos:   (B, N, 2) coordinates for all pixels (useful if needed later)
    """
    device = img_flat.device
    B, N, C = img_flat.shape
    assert N == img_size ** 2, "img_flat second dim must be img_size**2"

    # ----- build coordinates for all pixels -----
    pixel_idx = torch.arange(N, device=device)  # [0..N-1]

    # row (y) and col (x) indices
    x1 = pixel_idx // img_size  # (N,)
    x2 = pixel_idx % img_size   # (N,)

    # normalize to [-1, 1]
    pos = torch.stack([
        2.0 * x1.float() / (img_size - 1) - 1.0,  # y
        2.0 * x2.float() / (img_size - 1) - 1.0,  # x
    ], dim=-1)  # (N, 2)
    pos = pos.unsqueeze(0).expand(B, -1, -1)      # (B, N, 2)

    # ----- choose sizes for ctx and tgt -----
    max_ctx = int(N * max_ctx_frac)
    ctx_size = torch.randint(low=min_ctx, high=max_ctx, size=(1,), device=device).item()
    tgt_size = int(N * tgt_frac)

    # ----- random permutation per image -----
    # each row of 'idxs' is a random permutation of [0..N-1]
    idxs = torch.rand(B, N, device=device).argsort(dim=-1)

    ctx_idxs = idxs[..., :ctx_size]                     # (B, ctx_size)
    tgt_idxs = idxs[..., ctx_size // 2 : tgt_size]      # partial overlap like your code
    # effective target size:
    tgt_size_eff = tgt_idxs.shape[-1]

    batch_idx = torch.arange(B, device=device).unsqueeze(-1)  # (B, 1)

    # ----- gather coords & values -----
    ctx_x = pos[batch_idx, ctx_idxs]        # (B, ctx_size, 2)
    ctx_y = img_flat[batch_idx, ctx_idxs]   # (B, ctx_size, C)

    tgt_x = pos[batch_idx, tgt_idxs]        # (B, tgt_size_eff, 2)
    tgt_y = img_flat[batch_idx, tgt_idxs]   # (B, tgt_size_eff, C)

    ctx_tgt_xy = dict(ctx_x=ctx_x, ctx_y=ctx_y, tgt_x=tgt_x, tgt_y=tgt_y)

    return ctx_tgt_xy, pos


def _coords_to_indices(coords: torch.Tensor, img_size: int):
    """
    coords: (..., 2) in [-1,1], order [y, x]
    returns: (y_idx, x_idx) in [0, img_size-1] as long tensors
    """
    # map [-1,1] -> [0, img_size-1]
    y = (coords[..., 0] + 1.0) * 0.5 * (img_size - 1)
    x = (coords[..., 1] + 1.0) * 0.5 * (img_size - 1)

    y_idx = y.round().long().clamp(0, img_size - 1)
    x_idx = x.round().long().clamp(0, img_size - 1)
    return y_idx, x_idx


def build_ctx_tgt_viz_images(
    img_flat: torch.Tensor,
    ctx_x: torch.Tensor,
    ctx_y: torch.Tensor,
    tgt_x: torch.Tensor,
    pred_y: torch.Tensor,   # y1_pred, same thing
    img_size: int,
    n_examples: int = 4,
):
    """
    Build an image tensor suitable for vutils.make_grid to show:

      Row 1: original images
      Row 2: context-only (masked) images
      Row 3: context + predicted targets

    Args:
        img_flat: (B, N, C), N = img_size**2, values in [-1, 1]
        ctx_x:    (B, ctx_size, 2)   normalized coords in [-1,1]^2
        ctx_y:    (B, ctx_size, C)
        tgt_x:    (B, tgt_size, 2)
        pred_y:   (B, tgt_size, C)
        img_size: H = W
        n_examples: how many batch items (columns) to visualize

    Returns:
        imgs: (3 * n_examples, C, H, W) tensor ready for make_grid
              order is:
                [orig_1..orig_N, ctx_1..ctx_N, ctx+pred_1..ctx+pred_N]
              so with nrow=n_examples you'll get 3 rows.
    """
    B, N, C = img_flat.shape
    assert N == img_size ** 2, "img_flat second dim must be img_size**2"
    n_examples = min(n_examples, B)

    orig_list = []
    ctx_list = []
    pred_list = []

    for b in range(n_examples):
        # ----- original (H, W, C) -----
        img_b = img_flat[b].reshape(img_size, img_size, C)  # (H, W, C)

        # ----- context-only image -----
        ctx_img = torch.ones_like(img_b)  # white canvas in [-1,1]
        ctx_y_idx, ctx_x_idx = _coords_to_indices(ctx_x[b], img_size)
        ctx_img[ctx_y_idx, ctx_x_idx] = ctx_y[b]

        # ----- context + predictions image -----
        # start from context-only canvas, then paint predicted targets
        pred_img = ctx_img.clone()
        tgt_y_idx, tgt_x_idx = _coords_to_indices(tgt_x[b], img_size)
        pred_img[tgt_y_idx, tgt_x_idx] = pred_y[b]

        # convert to (C, H, W)
        orig_list.append(img_b.permute(2, 0, 1))
        ctx_list.append(ctx_img.permute(2, 0, 1))
        pred_list.append(pred_img.permute(2, 0, 1))

    # order: all originals, then all ctx, then all ctx+pred
    imgs = torch.stack(orig_list + ctx_list + pred_list, dim=0)  # (3*n_examples, C, H, W)
    return imgs


def sample_ctx_tgt_test(img_flat: torch.Tensor, img_size: int, ctx_frac: float = 0.1):
    """
    Test-time split where context + target = all pixels, no overlap.

    img_flat: (B, N, C), N = img_size**2
    """
    device = img_flat.device
    B, N, C = img_flat.shape
    assert N == img_size**2

    # coords for all pixels, same as before
    pixel_idx = torch.arange(N, device=device)
    x1 = pixel_idx // img_size
    x2 = pixel_idx % img_size
    pos = torch.stack([
        2.0 * x1.float() / (img_size - 1) - 1.0,
        2.0 * x2.float() / (img_size - 1) - 1.0,
    ], dim=-1)          # (N, 2)
    pos = pos.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)

    # choose context size
    ctx_size = int(N * ctx_frac)

    # random permutation per image
    idxs = torch.rand(B, N, device=device).argsort(dim=-1)  # (B, N)

    ctx_idxs = idxs[..., :ctx_size]      # (B, ctx_size)

    # target = complement of context
    all_idxs = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
    mask = torch.ones_like(all_idxs, dtype=torch.bool)
    mask.scatter_(1, ctx_idxs, False)
    tgt_idxs = all_idxs[mask].view(B, N - ctx_size)  # (B, N - ctx_size)

    batch_idx = torch.arange(B, device=device).unsqueeze(-1)

    ctx_x = pos[batch_idx, ctx_idxs]         # (B, ctx_size, 2)
    ctx_y = img_flat[batch_idx, ctx_idxs]    # (B, ctx_size, C)

    tgt_x = pos[batch_idx, tgt_idxs]         # (B, N - ctx_size, 2)
    tgt_y = img_flat[batch_idx, tgt_idxs]    # if you want GT for eval

    ctx_tgt_xy = dict(ctx_x=ctx_x, ctx_y=ctx_y, tgt_x=tgt_x, tgt_y=tgt_y)

    return ctx_tgt_xy, pos
