from __future__ import annotations

import math
import time
from typing import Literal, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csgraph, csr_matrix

from vroc.common_types import Number


def filter1D(img, weight, dim, padding_mode="replicate"):
    B, C, D, H, W = img.shape
    N = weight.shape[0]

    padding = torch.zeros(
        6,
    )
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N // 2
    padding = padding.long().tolist()

    view = torch.ones(
        5,
        dtype=torch.float16,
    )
    view[dim + 2] = -1
    view = view.long().tolist()

    return F.conv3d(
        F.pad(img.view(B * C, 1, D, H, W), padding, mode=padding_mode),
        weight.view(view),
    ).view(B, C, D, H, W)


def smooth(img, sigma):
    device = img.device

    sigma = torch.tensor([sigma], device=device, dtype=torch.float16)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1

    weight = torch.exp(
        -torch.pow(
            torch.linspace(-(N // 2), N // 2, N, device=device, dtype=torch.float32), 2
        )
        / (2 * torch.pow(sigma, 2))
    )
    weight /= weight.sum()

    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)

    return img


def structure_tensor(img, sigma):
    B, C, D, H, W = img.shape

    struct = []
    for i in range(C):
        for j in range(i, C):
            struct.append(smooth((img[:, i, ...] * img[:, j, ...]).unsqueeze(1), sigma))

    return torch.cat(struct, dim=1)


def invert_structure_tensor(struct):
    a = struct[:, 0, ...]
    b = struct[:, 1, ...]
    c = struct[:, 2, ...]
    e = struct[:, 3, ...]
    f = struct[:, 4, ...]
    i = struct[:, 5, ...]

    A = e * i - f * f
    B = -b * i + c * f
    C = b * f - c * e
    E = a * i - c * c
    F = -a * f + b * c
    I = a * e - b * b

    det = (a * A + b * B + c * C).unsqueeze(1)

    struct_inv = (1.0 / det) * torch.stack([A, B, C, E, F, I], dim=1)

    return struct_inv


def kpts_pt(kpts_world, shape, align_corners=None):
    device = kpts_world.device
    D, H, W = shape

    kpts_pt_ = (
        kpts_world.flip(-1) / (torch.tensor([W, H, D], device=device) - 1)
    ) * 2 - 1
    if not align_corners:
        kpts_pt_ *= (torch.tensor([W, H, D], device=device) - 1) / torch.tensor(
            [W, H, D], device=device
        )

    return kpts_pt_


def kpts_world(kpts_pt, shape, align_corners=None):
    device = kpts_pt.device
    D, H, W = shape

    if not align_corners:
        kpts_pt /= (torch.tensor([W, H, D], device=device) - 1) / torch.tensor(
            [W, H, D], device=device
        )
    kpts_world_ = (
        ((kpts_pt + 1) / 2) * (torch.tensor([W, H, D], device=device) - 1)
    ).flip(-1)

    return kpts_world_


def flow_world(flow_pt, shape, align_corners=None):
    device = flow_pt.device
    D, H, W = shape

    if not align_corners:
        flow_pt /= (torch.tensor([W, H, D], device=device) - 1) / torch.tensor(
            [W, H, D], device=device
        )
    flow_world_ = ((flow_pt / 2) * (torch.tensor([W, H, D], device=device) - 1)).flip(
        -1
    )

    return flow_world_


def foerstner_kpts(img, mask, sigma=1.4, d=9, thresh=1e-8):
    _, _, D, H, W = img.shape
    device = img.device

    filt = torch.tensor(
        [1.0 / 12.0, -8.0 / 12.0, 0.0, 8.0 / 12.0, -1.0 / 12.0],
        device=device,
        dtype=torch.float32,
    )
    grad = torch.cat(
        [filter1D(img, filt, 0), filter1D(img, filt, 1), filter1D(img, filt, 2)], dim=1
    )

    struct_inv = invert_structure_tensor(structure_tensor(grad, sigma))

    distinctiveness = 1.0 / (
        struct_inv[:, 0, ...] + struct_inv[:, 3, ...] + struct_inv[:, 5, ...]
    ).unsqueeze(1)

    pad1 = d // 2
    pad2 = d - pad1 - 1

    maxfeat = F.max_pool3d(
        F.pad(distinctiveness, (pad2, pad1, pad2, pad1, pad2, pad1)), d, stride=1
    )

    structure_element = torch.tensor(
        [
            [[0.0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ],
        device=device,
    )

    mask_eroded = (
        1
        - F.conv3d(
            1 - mask.float(), structure_element.unsqueeze(0).unsqueeze(0), padding=1
        ).clamp_(0, 1)
    ).bool()

    if isinstance(thresh, float):
        # use thresh as foerstner threshold
        kpts = torch.nonzero(
            mask_eroded & (maxfeat == distinctiveness) & (distinctiveness >= thresh)
        ).unsqueeze(0)[:, :, 2:]

    elif isinstance(thresh, int):
        # return top k keypoints, with k = threh
        kpts = torch.nonzero(mask_eroded & (maxfeat == distinctiveness)).unsqueeze(0)[
            :, :, 2:
        ]
        # sort keypoints based from high to low distinctiveness
        distinctiveness_of_kpts = distinctiveness[
            0, 0, kpts[0, :, 0], kpts[0, :, 1], kpts[0, :, 2]
        ]
        sorted_indices = torch.argsort(distinctiveness_of_kpts, descending=True)
        sorted_indices = sorted_indices[:thresh]
        kpts = kpts[:, sorted_indices]
    else:
        raise ValueError("thresh must be either float or int")

    return kpts_pt(kpts, (D, H, W), align_corners=True)


def mindssc(img, delta=1, sigma=1):
    device = img.device

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.tensor(
        [[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]],
        dtype=torch.float32,
        device=device,
    )

    # squared distances
    dist = pdist(six_neighbourhood.unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(
        torch.arange(6, device=device, dtype=torch.float32),
        torch.arange(6, device=device, dtype=torch.float32),
    )
    mask = (x > y).view(-1) & (dist == 2).view(-1)

    # build kernel
    idx_shift1 = (
        six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :].long()
    )
    idx_shift2 = (
        six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :].long()
    )
    mshift1 = torch.zeros((12, 1, 3, 3, 3), device=device, dtype=torch.float32)
    mshift1.view(-1)[
        torch.arange(12, device=device) * 27
        + idx_shift1[:, 0] * 9
        + idx_shift1[:, 1] * 3
        + idx_shift1[:, 2]
    ] = 1
    mshift2 = torch.zeros((12, 1, 3, 3, 3), device=device, dtype=torch.float32)
    mshift2.view(-1)[
        torch.arange(12, device=device) * 27
        + idx_shift2[:, 0] * 9
        + idx_shift2[:, 1] * 3
        + idx_shift2[:, 2]
    ] = 1
    rpad = nn.ReplicationPad3d(delta)

    # compute patch-ssd
    ssd = smooth(
        (
            (
                F.conv3d(rpad(img), mshift1, dilation=delta)
                - F.conv3d(rpad(img), mshift2, dilation=delta)
            )
            ** 2
        ),
        sigma,
    )

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
    mind /= mind_var
    mind = torch.exp(-mind)

    return mind


def minimum_spanning_tree(dist):
    device = dist.device
    N = dist.shape[1]

    mst = csgraph.minimum_spanning_tree(csr_matrix(dist[0].cpu().numpy()))
    bfo = csgraph.breadth_first_order(mst, 0, directed=False)
    edges = (
        torch.tensor([bfo[1][bfo[0]][1:], bfo[0][1:]], dtype=torch.long, device=device)
        .t()
        .view(1, -1, 2)
    )

    level = torch.zeros((1, N, 1), dtype=torch.long, device=device)
    for i in range(N - 1):
        level[0, edges[0, i, 1], 0] = level[0, edges[0, i, 0], 0] + 1

    idx = edges[0, :, 1].sort()[1]
    edges = edges[:, idx, :]

    return edges, level


def minconv(input):
    device = input.device
    disp_width = input.shape[-1]

    disp1d = torch.linspace(
        -(disp_width // 2), disp_width // 2, disp_width, device=device
    )
    regular1d = (disp1d.view(1, -1) - disp1d.view(-1, 1)) ** 2

    output = torch.min(
        input.view(-1, disp_width, 1, disp_width, disp_width)
        + regular1d.view(1, disp_width, disp_width, 1, 1),
        1,
    )[0]
    output = torch.min(
        output.view(-1, disp_width, disp_width, 1, disp_width)
        + regular1d.view(1, 1, disp_width, disp_width, 1),
        2,
    )[0]
    output = torch.min(
        output.view(-1, disp_width, disp_width, disp_width, 1)
        + regular1d.view(1, 1, 1, disp_width, disp_width),
        3,
    )[0]

    output = output - (torch.min(output.view(-1, disp_width**3), 1)[0]).view(
        -1, 1, 1, 1
    )

    return output.view_as(input)


def tbp(cost, edges, level, dist):
    marginals = cost
    message = torch.zeros_like(marginals)

    for i in range(level.max(), 0, -1):
        child = edges[0, level[0, 1:, 0] == i, 1]
        parent = edges[0, level[0, 1:, 0] == i, 0]
        weight = dist[0, child, parent].view(-1, 1, 1, 1)

        data = marginals[:, child, :, :, :]
        data_reg = minconv(data * weight) / weight

        message[:, child, :, :, :] = data_reg
        marginals = torch.index_add(marginals, 1, parent, data_reg)

    for i in range(1, level.max() + 1):
        child = edges[0, level[0, 1:, 0] == i, 1]
        parent = edges[0, level[0, 1:, 0] == i, 0]
        weight = dist[0, child, parent].view(-1, 1, 1, 1)

        data = (
            marginals[:, parent, :, :, :]
            - message[:, child, :, :, :]
            + message[:, parent, :, :, :]
        )
        data_reg = minconv(data * weight) / weight

        message[:, child, :, :, :] = data_reg

    marginals += message

    return marginals


def mean_filter(img, r):
    device = img.device

    weight = torch.ones((2 * r + 1,), device=device, dtype=torch.float32) / (2 * r + 1)

    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)

    return img


def pdist(x, p=2):
    if p == 1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=3)
    elif p == 2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist


def kpts_dist(kpts, img, beta, k=64):
    device = kpts.device
    B, N, _ = kpts.shape
    _, _, D, H, W = img.shape

    dist = pdist(kpts_world(kpts, (D, H, W), align_corners=True)).sqrt()
    dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 1e15
    dist[dist < 0.1] = 0.1
    img_mean = mean_filter(img, 2)
    kpts_mean = F.grid_sample(
        img_mean,
        kpts.view(1, 1, 1, -1, 3).to(img_mean.dtype),
        mode="nearest",
        align_corners=True,
    ).view(1, -1, 1)
    dist += pdist(kpts_mean, p=1) / beta

    include_self = False
    ind = (-dist).topk(k + (1 - int(include_self)), dim=-1)[1][
        :, :, 1 - int(include_self) :
    ]
    A = torch.zeros((B, N, N), device=device)
    A[:, torch.arange(N).repeat(k), ind[0].t().contiguous().view(-1)] = 1
    A[:, ind[0].t().contiguous().view(-1), torch.arange(N).repeat(k)] = 1
    dist = A * dist

    return dist


def get_patch(patch_step, patch_radius, shape, device):
    D, H, W = shape

    patch = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, 2 * patch_radius + 1, patch_step, device=device),
                torch.arange(0, 2 * patch_radius + 1, patch_step, device=device),
                torch.arange(0, 2 * patch_radius + 1, patch_step, device=device),
            ),
            dim=3,
        ).view(1, -1, 3)
        - patch_radius
    )
    patch = flow_pt(patch, (D, H, W), align_corners=True)
    return patch


def ssd(
    kpts_fixed,
    feat_fixed,
    feat_moving,
    disp_radius=16,
    disp_step=2,
    patch_radius=3,
    unroll_step_size=2**6,
):
    device = kpts_fixed.device
    N = kpts_fixed.shape[1]
    _, C, D, H, W = feat_fixed.shape

    patch_step = disp_step  # same stride necessary for fast implementation
    patch = get_patch(patch_step, patch_radius, (D, H, W), device=device)
    patch_width = round(patch.shape[1] ** (1.0 / 3))

    pad = [(patch_width - 1) // 2, (patch_width - 1) // 2 + (1 - patch_width % 2)]

    disp = get_disp(
        disp_step, disp_radius + ((pad[0] + pad[1]) / 2), (D, H, W), device=device
    )
    disp_width = disp_radius * 2 + 1

    cost = torch.zeros(1, N, disp_width, disp_width, disp_width, device=device)
    n = math.ceil(N / unroll_step_size)
    for j in range(n):
        j1 = j * unroll_step_size
        j2 = min((j + 1) * unroll_step_size, N)

        feat_fixed_patch = F.grid_sample(
            feat_fixed,
            kpts_fixed[:, j1:j2, :].view(1, -1, 1, 1, 3) + patch.view(1, 1, -1, 1, 3),
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        )
        feat_moving_disp = F.grid_sample(
            feat_moving,
            kpts_fixed[:, j1:j2, :].view(1, -1, 1, 1, 3) + disp.view(1, 1, -1, 1, 3),
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        )

        fixed_sum = (feat_fixed_patch**2).sum(dim=3).view(C, (j2 - j1), 1, 1, 1)
        moving_sum = (patch_width**3) * F.avg_pool3d(
            (feat_moving_disp**2).view(
                C,
                -1,
                disp_width + pad[0] + pad[1],
                disp_width + pad[0] + pad[1],
                disp_width + pad[0] + pad[1],
            ),
            patch_width,
            stride=1,
        ).view(C, (j2 - j1), disp_width, disp_width, disp_width)
        corr = F.conv3d(
            feat_moving_disp.view(
                1,
                -1,
                disp_width + pad[0] + pad[1],
                disp_width + pad[0] + pad[1],
                disp_width + pad[0] + pad[1],
            ),
            feat_fixed_patch.view(-1, 1, patch_width, patch_width, patch_width),
            groups=C * (j2 - j1),
        ).view(C, (j2 - j1), disp_width, disp_width, disp_width)

        cost[0, j1:j2, :, :, :] = (fixed_sum + moving_sum - 2 * corr).sum(dim=0) / (
            patch_width**3
        )

    return cost


def compute_marginals(
    kpts_fix,
    img_fix,
    mind_fix,
    mind_mov,
    alpha,
    beta,
    disp_radius,
    disp_step,
    patch_radius,
):
    cost = alpha * ssd(
        kpts_fix, mind_fix, mind_mov, disp_radius, disp_step, patch_radius
    )

    dist = kpts_dist(kpts_fix, img_fix, beta)
    edges, level = minimum_spanning_tree(dist)
    marginals = tbp(cost, edges, level, dist)

    return marginals


def flow_pt(flow_world, shape, align_corners=None):
    device = flow_world.device
    D, H, W = shape

    flow_pt_ = (flow_world.flip(-1) / (torch.tensor([W, H, D], device=device) - 1)) * 2
    if not align_corners:
        flow_pt_ *= (torch.tensor([W, H, D], device=device) - 1) / torch.tensor(
            [W, H, D], device=device
        )

    return flow_pt_


def get_disp(disp_step, disp_radius, shape, device):
    D, H, W = shape

    disp = torch.stack(
        torch.meshgrid(
            torch.arange(
                -disp_step * disp_radius,
                disp_step * disp_radius + 1,
                disp_step,
                device=device,
            ),
            torch.arange(
                -disp_step * disp_radius,
                disp_step * disp_radius + 1,
                disp_step,
                device=device,
            ),
            torch.arange(
                -disp_step * disp_radius,
                disp_step * disp_radius + 1,
                disp_step,
                device=device,
            ),
        ),
        dim=3,
    ).view(1, -1, 3)

    disp = flow_pt(disp, (D, H, W), align_corners=True)
    return disp


def find_rigid_3d(x, y):
    device = x.device
    x_mean = x[:, :3].mean(0)
    y_mean = y[:, :3].mean(0)
    u, s, v = torch.svd(torch.matmul((x[:, :3] - x_mean).t(), (y[:, :3] - y_mean)))
    m = torch.eye(v.shape[0], v.shape[0], device=device)
    m[-1, -1] = torch.det(torch.matmul(v, u.t()))
    rotation = torch.matmul(torch.matmul(v, m), u.t())
    translation = y_mean - torch.matmul(rotation, x_mean)
    T = torch.eye(4, device=device)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def compute_rigid_transform(kpts_fixed, kpts_moving, iter=5):
    device = kpts_fixed.device
    kpts_fixed = torch.cat(
        (kpts_fixed, torch.ones(1, kpts_fixed.shape[1], 1, device=device)), 2
    )
    kpts_moving = torch.cat(
        (kpts_moving, torch.ones(1, kpts_moving.shape[1], 1, device=device)), 2
    )
    idx = torch.arange(kpts_fixed.shape[1]).to(kpts_fixed.device)[
        torch.randperm(kpts_fixed.shape[1])[: kpts_fixed.shape[1] // 2]
    ]
    for i in range(iter):
        x = find_rigid_3d(kpts_fixed[0, idx, :], kpts_moving[0, idx, :]).t()
        residual = torch.sqrt(
            torch.sum(torch.pow(kpts_moving[0] - torch.mm(kpts_fixed[0], x), 2), 1)
        )
        _, idx = torch.topk(residual, kpts_fixed.shape[1] // 2, largest=False)
    return x.t().unsqueeze(0)


class TPS:
    @staticmethod
    def fit(c, f, lambd=0.0):
        device = c.device

        n = c.shape[0]
        f_dim = f.shape[1]

        U = TPS.u(TPS.d(c, c))
        K = U + torch.eye(n, device=device) * lambd

        P = torch.ones((n, 4), device=device)
        P[:, 1:] = c

        v = torch.zeros((n + 4, f_dim), device=device)
        v[:n, :] = f

        A = torch.zeros((n + 4, n + 4), device=device)
        A[:n, :n] = K
        A[:n, -4:] = P
        A[-4:, :n] = P.t()

        theta = torch.linalg.solve(A, v)

        return theta

    @staticmethod
    def d(a, b):
        ra = (a**2).sum(dim=1).view(-1, 1)
        rb = (b**2).sum(dim=1).view(1, -1)
        dist = ra + rb - 2.0 * torch.mm(a, b.permute(1, 0))
        dist.clamp_(0.0, float("inf"))
        return torch.sqrt(dist)

    @staticmethod
    def u(r):
        return (r**2) * torch.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-4], theta[-4:].unsqueeze(2)
        b = torch.matmul(U, w)
        return (a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + a[3] * x[:, 2] + b.t()).t()


def thin_plate_dense(x1, y1, shape, step, lambd=0.0, unroll_step_size=2**12):
    device = x1.device
    D, H, W = shape
    D1, H1, W1 = D // step, H // step, W // step

    x2 = F.affine_grid(
        torch.eye(3, 4, device=device).unsqueeze(0),
        (1, 1, D1, H1, W1),
        align_corners=True,
    ).view(-1, 3)
    tps = TPS()
    theta = tps.fit(x1[0], y1[0], lambd)

    y2 = torch.zeros((1, D1 * H1 * W1, 3), device=device)
    N = D1 * H1 * W1
    n = math.ceil(N / unroll_step_size)
    for j in range(n):
        j1 = j * unroll_step_size
        j2 = min((j + 1) * unroll_step_size, N)
        y2[0, j1:j2, :] = tps.z(x2[j1:j2], x1[0], theta)

    y2 = y2.view(1, D1, H1, W1, 3).permute(0, 4, 1, 2, 3)
    y2 = F.interpolate(y2, (D, H, W), mode="trilinear", align_corners=True).permute(
        0, 2, 3, 4, 1
    )

    return y2


def extract_keypoints(
    moving_image: torch.Tensor,
    fixed_image: torch.Tensor,
    fixed_mask: torch.Tensor,
    # alpha: Number = 2.5,
    # beta: Number = 150,
    # gamma: Number = 5,
    # delta: Number = 1,
    # lambd: Number = 0,
    # sigma_foerstner: Number = 1.4,
    # sigma_mind: Number = 0.8,
    # search_radius: Sequence[float] = (16, 8),
    # length: Sequence[float] = (6, 3),
    # quantization: Sequence[float] = (2, 1),
    # patch_radius: Sequence[float] = (3, 2),
    # transform: Sequence[Literal["rigid", "dense"]] = ("dense", "dense"),
    alpha: Number = 2.5,
    beta: Number = 150,
    gamma: Number = 5,
    delta: Number = 1,
    lambd: Number = 0,
    sigma_foerstner: Number = 1.4,
    keypoint_treshold: float | int = 300,
    sigma_mind: Number = 0.8,
    search_radius: Sequence[float] = (24, 8),
    length: Sequence[float] = (6, 3),
    quantization: Sequence[float] = (2, 1),
    patch_radius: Sequence[float] = (3, 2),
    transform: Sequence[Literal["rigid", "dense"]] = ("rigid", "dense"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        device = fixed_image.device
        _, _, D, H, W = fixed_image.shape

        # print("Compute fixed MIND features ...", end=" ")
        torch.cuda.synchronize()
        t0 = time.time()
        mind_fix = mindssc(fixed_image, delta, sigma_mind)
        torch.cuda.synchronize()
        t1 = time.time()
        # print("finished ({:.2f} s).".format(t1 - t0))

        dense_flow = torch.zeros((1, D, H, W, 3), device=device)
        img_mov_warped = moving_image
        for i in range(len(search_radius)):
            # print("Stage {}/{}".format(i + 1, len(search_radius)))
            # print("    search radius: {}".format(search_radius[i]))
            # print("      cube length: {}".format(length[i]))
            # print("     quantisation: {}".format(quantization[i]))
            # print("     patch radius: {}".format(patch_radius[i]))
            # print("        transform: {}".format(transform[i]))

            disp = get_disp(quantization[i], search_radius[i], (D, H, W), device=device)

            # print("    Compute moving MIND features ...", end=" ")
            torch.cuda.synchronize()
            t0 = time.time()
            mind_mov = mindssc(img_mov_warped, delta, sigma_mind)
            torch.cuda.synchronize()
            t1 = time.time()
            # print("finished ({:.2f} s).".format(t1 - t0))

            torch.cuda.synchronize()
            t0 = time.time()
            kpts_fix = foerstner_kpts(
                fixed_image,
                fixed_mask,
                sigma_foerstner,
                length[i],
                thresh=keypoint_treshold,
            )
            torch.cuda.synchronize()
            t1 = time.time()
            # print(
            #     "    {} fixed keypoints extracted ({:.2f} s).".format(
            #         kpts_fix.shape[1], t1 - t0
            #     )
            # )

            # print("    Compute forward marginals ...", end=" ")
            torch.cuda.synchronize()
            t0 = time.time()
            marginalsf = compute_marginals(
                kpts_fix,
                fixed_image,
                mind_fix,
                mind_mov,
                alpha,
                beta,
                search_radius[i],
                quantization[i],
                patch_radius[i],
            )
            torch.cuda.synchronize()
            t1 = time.time()
            # print("finished ({:.2f} s).".format(t1 - t0))

            flow = (
                F.softmax(-gamma * marginalsf.view(1, kpts_fix.shape[1], -1, 1), dim=2)
                * disp.view(1, 1, -1, 3)
            ).sum(2)

            kpts_mov = kpts_fix + flow

            # print("    Compute symmetric backward marginals ...", end=" ")
            torch.cuda.synchronize()
            t0 = time.time()
            marginalsb = compute_marginals(
                kpts_mov,
                fixed_image,
                mind_mov,
                mind_fix,
                alpha,
                beta,
                search_radius[i],
                quantization[i],
                patch_radius[i],
            )
            torch.cuda.synchronize()
            t1 = time.time()
            # print("finished ({:.2f} s).".format(t1 - t0))

            marginals = 0.5 * (
                marginalsf.view(1, kpts_fix.shape[1], -1)
                + marginalsb.view(1, kpts_fix.shape[1], -1).flip(2)
            )

            flow = (
                F.softmax(-gamma * marginals.view(1, kpts_fix.shape[1], -1, 1), dim=2)
                * disp.view(1, 1, -1, 3)
            ).sum(2)

            torch.cuda.synchronize()
            t0 = time.time()
            if transform[i] == "rigid":
                # print("    Find rigid transform ...", end=" ")
                rigid = compute_rigid_transform(kpts_fix, kpts_fix + flow)
                dense_flow_ = F.affine_grid(
                    rigid[:, :3, :] - torch.eye(3, 4, device=device).unsqueeze(0),
                    (1, 1, D, H, W),
                    align_corners=True,
                )
            elif transform[i] == "dense":
                # print("    Dense thin plate spline interpolation ...", end=" ")
                dense_flow_ = thin_plate_dense(kpts_fix, flow, (D, H, W), 2, lambd)
            torch.cuda.synchronize()
            t1 = time.time()
            # print("finished ({:.2f} s).".format(t1 - t0))

            dense_flow += dense_flow_

            img_mov_warped = F.grid_sample(
                moving_image,
                F.affine_grid(
                    torch.eye(3, 4, dtype=moving_image.dtype, device=device).unsqueeze(
                        0
                    ),
                    (1, 1, D, H, W),
                    align_corners=True,
                )
                + dense_flow.to(moving_image.dtype),
                align_corners=True,
            )

        flow = (
            F.grid_sample(
                dense_flow.permute(0, 4, 1, 2, 3),
                kpts_fix.view(1, 1, 1, -1, 3),
                align_corners=True,
            )
            .view(1, 3, -1)
            .permute(0, 2, 1)
        )

        return (
            kpts_world(kpts_fix + flow, (D, H, W), align_corners=True),
            kpts_world(kpts_fix, (D, H, W), align_corners=True),
            flow_world(dense_flow.view(1, -1, 3), (D, H, W), align_corners=True).view(
                1, D, H, W, 3
            ),
            img_mov_warped,
        )
