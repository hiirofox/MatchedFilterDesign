import os
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    src_dir: str = r"D:\Projects\Py\matchedfilterdesign\iirCoeffsLuts"
    lut_file: str = "iirluts2b.txt"
    save_file: str = "structured_transport_best.pt"

    fs: float = 48000.0
    f_min_resp: float = 70.0
    num_freq: int = 768

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1234

    hidden: int = 48
    batch_size: int = 128
    steps_per_epoch: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-6

    # base selection
    neighbor_pool: int = 32
    max_base_param_dist: float = 0.45

    # loss weights
    w_proto: float = 1.0
    w_lut_resp: float = 0.6
    w_move: float = 0.01
    w_smooth: float = 0.02

    # parameter perturbation for smoothness
    delta_log_fc: float = 0.018
    delta_log_q: float = 0.018
    delta_gain: float = 0.35
    delta_stages: float = 0.05

    # output residual scales
    dr_scale: float = 0.18
    dtheta_scale: float = 0.22
    dg_scale: float = 0.25

    print_every: int = 1
    save_every: int = 1


CFG = Config()


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_log_freq_grid(fs: float, f_min_resp: float, num_freq: int):
    w_min = 2.0 * np.pi * f_min_resp / fs
    w_grid = np.logspace(np.log10(w_min), np.log10(np.pi), num_freq).astype(np.float32)
    freqs = w_grid * fs / (2.0 * np.pi)
    return w_grid, freqs


def db_torch(x: torch.Tensor, eps: float = 1e-9):
    return 20.0 * torch.log10(x + eps)


# ============================================================
# Parse LUT
# ============================================================

def parse_lut_file(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 19:
                continue

            vals = list(map(float, parts))
            dims = np.array(vals[0:4], dtype=np.int32)
            proto = np.array(vals[4:8], dtype=np.float32)   # [fc,Q,gain,stages]
            err = float(vals[8])
            b = np.array(vals[9:14], dtype=np.float32)
            a = np.array(vals[14:19], dtype=np.float32)

            rows.append({
                "dims": dims,
                "proto": proto,
                "err": err,
                "b": b,
                "a": a,
            })

    if not rows:
        raise RuntimeError(f"No valid rows found in {path}")
    return rows


# ============================================================
# Parameter normalizer
# ============================================================

class ParamNormalizer:
    def __init__(self, rows):
        params = np.stack([r["proto"] for r in rows], axis=0)
        self.fc_min = float(np.min(params[:, 0]))
        self.fc_max = float(np.max(params[:, 0]))
        self.q_min = float(np.min(params[:, 1]))
        self.q_max = float(np.max(params[:, 1]))
        self.g_min = float(np.min(params[:, 2]))
        self.g_max = float(np.max(params[:, 2]))
        self.s_min = float(np.min(params[:, 3]))
        self.s_max = float(np.max(params[:, 3]))

        self.log_fc_min = math.log(self.fc_min)
        self.log_fc_max = math.log(self.fc_max)
        self.log_q_min = math.log(self.q_min)
        self.log_q_max = math.log(self.q_max)

    def encode_np(self, p):
        fc, q, g, s = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        x0 = (np.log(fc) - self.log_fc_min) / (self.log_fc_max - self.log_fc_min) * 2.0 - 1.0
        x1 = (np.log(q) - self.log_q_min) / (self.log_q_max - self.log_q_min) * 2.0 - 1.0
        x2 = (g - self.g_min) / (self.g_max - self.g_min) * 2.0 - 1.0
        x3 = (s - self.s_min) / (self.s_max - self.s_min) * 2.0 - 1.0
        return np.stack([x0, x1, x2, x3], axis=-1).astype(np.float32)

    def encode_torch(self, p: torch.Tensor):
        fc, q, g, s = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        x0 = (torch.log(fc) - self.log_fc_min) / (self.log_fc_max - self.log_fc_min) * 2.0 - 1.0
        x1 = (torch.log(q) - self.log_q_min) / (self.log_q_max - self.log_q_min) * 2.0 - 1.0
        x2 = (g - self.g_min) / (self.g_max - self.g_min) * 2.0 - 1.0
        x3 = (s - self.s_min) / (self.s_max - self.s_min) * 2.0 - 1.0
        return torch.stack([x0, x1, x2, x3], dim=1)

    def clamp_torch(self, p: torch.Tensor):
        x = p.clone()
        x[:, 0] = x[:, 0].clamp(self.fc_min, self.fc_max)
        x[:, 1] = x[:, 1].clamp(self.q_min, self.q_max)
        x[:, 2] = x[:, 2].clamp(self.g_min, self.g_max)
        x[:, 3] = x[:, 3].clamp(self.s_min, self.s_max)
        return x


# ============================================================
# Prototype teacher
# ============================================================

def prototype_mag_response_torch(params: torch.Tensor, freqs_hz: torch.Tensor) -> torch.Tensor:
    fc = params[:, 0:1]
    Q = params[:, 1:2]
    gain_db = params[:, 2:3]
    stages = params[:, 3:4]

    wc = 2.0 * math.pi * fc
    omega = 2.0 * math.pi * freqs_hz[None, :]
    A_target = torch.pow(10.0, gain_db / 20.0)

    num = torch.abs((omega ** 2 - wc ** 2) / (wc / (Q + 1e-12) * omega + 1e-200))
    denom = 1.0 + torch.pow(num + 1e-30, 2.0 * stages)
    mag = 1.0 + (A_target - 1.0) * (1.0 / denom)
    return mag


# ============================================================
# IIR response
# ============================================================

def iir_mag_response_torch(b: torch.Tensor, a: torch.Tensor, w_grid: torch.Tensor):
    k = torch.arange(5, device=b.device, dtype=w_grid.dtype)
    ejwk = torch.exp(-1j * w_grid[:, None] * k[None, :])

    num = torch.sum(b[:, None, :] * ejwk[None, :, :], dim=-1)
    den = torch.sum(a[:, None, :] * ejwk[None, :, :], dim=-1)

    H = num / (den + 1e-12)
    return torch.abs(H)


# ============================================================
# Pole-zero extraction / reconstruction
# ============================================================
def select_upper_members_zplane(zroots: np.ndarray):
    """
    强制返回 2 个代表根。
    策略：
    1. 先取 imag > 0 的根
    2. 不够则补实根
    3. 还不够则从全部根里按“更像上半平面代表根”的顺序补齐
    4. 最终无论如何只返回 2 个
    """
    zroots = np.asarray(zroots, dtype=np.complex128).ravel()

    if zroots.size == 0:
        # 极端异常保护
        reps = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
    else:
        reps = []

        # 先取上半平面
        for z in zroots:
            if np.imag(z) > 1e-8:
                reps.append(z)

        # 不够则补实根
        if len(reps) < 2:
            real_roots = [z for z in zroots if abs(np.imag(z)) <= 1e-8]
            real_roots = sorted(real_roots, key=lambda x: abs(x))
            for z in real_roots:
                reps.append(complex(np.real(z), 0.0))
                if len(reps) >= 2:
                    break

        # 还不够则从全部根里补
        if len(reps) < 2:
            z_sorted = sorted(
                zroots,
                key=lambda x: (-np.imag(x), abs(x))
            )
            for z in z_sorted:
                reps.append(z)
                if len(reps) >= 2:
                    break

        reps = np.asarray(reps[:2], dtype=np.complex128)

        # 如果仍然不足 2 个，最后硬补
        if reps.size < 2:
            reps = np.pad(reps, (0, 2 - reps.size), constant_values=(1.0 + 0j))

    r = np.abs(reps)
    th = np.abs(np.angle(reps))

    # 固定排序
    idx = np.argsort(th)
    r = r[idx]
    th = th[idx]

    # 强制 shape=(2,)
    r = np.asarray(r[:2], dtype=np.float32).reshape(2,)
    th = np.asarray(th[:2], dtype=np.float32).reshape(2,)

    return r, th

def qpoly_to_zplane_rtheta(qpoly_asc: np.ndarray):
    """
    qpoly_asc: q=z^-1 升幂系数
    转 z 平面根，再提取 2 个代表根的 (r, theta)
    强制返回 shape=(2,), (2,)
    """
    qpoly_asc = np.asarray(qpoly_asc, dtype=np.float64).ravel()

    # 长度保护
    if qpoly_asc.size < 2:
        qpoly_asc = np.array([1.0, 0.0], dtype=np.float64)

    # roots 需要降幂
    qpoly_desc = qpoly_asc[::-1]

    try:
        qroots = np.roots(qpoly_desc)
    except Exception:
        qroots = np.array([], dtype=np.complex128)

    zroots = []
    for qr in np.asarray(qroots).ravel():
        if abs(qr) < 1e-12:
            zroots.append(complex(1e12, 0.0))
        else:
            zroots.append(1.0 / qr)

    zroots = np.asarray(zroots, dtype=np.complex128)
    r, th = select_upper_members_zplane(zroots)

    # 最终双保险
    r = np.asarray(r, dtype=np.float32).reshape(2,)
    th = np.asarray(th, dtype=np.float32).reshape(2,)
    return r, th

def zplane_rtheta_to_qpoly_asc(r: np.ndarray, th: np.ndarray):
    zroots = []
    for rk, tk in zip(r, th):
        zroots.append(rk * np.exp(1j * tk))
        zroots.append(rk * np.exp(-1j * tk))
    zroots = np.array(zroots, dtype=np.complex128)

    qroots = 1.0 / zroots
    qpoly_desc = np.poly(qroots)
    qpoly_asc = np.real_if_close(qpoly_desc[::-1]).astype(np.float64)

    if abs(qpoly_asc[0]) < 1e-12:
        qpoly_asc[0] = 1.0
    qpoly_asc = qpoly_asc / qpoly_asc[0]
    return np.real(qpoly_asc).astype(np.float32)

def extract_structured_repr(row):
    b = np.asarray(row["b"], dtype=np.float64).ravel()
    a = np.asarray(row["a"], dtype=np.float64).ravel()

    rz, thz = qpoly_to_zplane_rtheta(b)
    rp, thp = qpoly_to_zplane_rtheta(a)

    g0 = np.array([b[0] if b.size > 0 else 1.0], dtype=np.float32)

    vec = np.concatenate([
        rz.reshape(2,),
        thz.reshape(2,),
        rp.reshape(2,),
        thp.reshape(2,),
        g0.reshape(1,)
    ], axis=0).astype(np.float32)

    if vec.shape != (9,):
        raise RuntimeError(f"structured repr shape is {vec.shape}, expected (9,)")

    return vec

def reconstruct_from_struct_torch(struct_vec: torch.Tensor):
    """
    struct_vec: [B,9]
      [rz1, rz2, thz1, thz2, rp1, rp2, thp1, thp2, g0]
    return b,a in q-domain ascending coefficients.
    """
    rz1, rz2, thz1, thz2, rp1, rp2, thp1, thp2, g0 = torch.chunk(struct_vec, 9, dim=1)

    rz1 = rz1.squeeze(1)
    rz2 = rz2.squeeze(1)
    thz1 = thz1.squeeze(1)
    thz2 = thz2.squeeze(1)
    rp1 = rp1.squeeze(1)
    rp2 = rp2.squeeze(1)
    thp1 = thp1.squeeze(1)
    thp2 = thp2.squeeze(1)
    g0 = g0.squeeze(1)

    # z-plane root -> q-domain biquad:
    # roots in z: r e^{±jθ}
    # q roots = 1/z = (1/r)e^{∓jθ}
    # q-poly factor: 1 - 2*(1/r)cosθ q + (1/r^2) q^2
    def q_biquad_from_z_rtheta(r, th):
        invr = 1.0 / (r + 1e-12)
        c1 = -2.0 * invr * torch.cos(th)
        c2 = invr * invr
        return c1, c2

    bz1, bz2 = q_biquad_from_z_rtheta(rz1, thz1)
    bz3, bz4 = q_biquad_from_z_rtheta(rz2, thz2)
    ap1, ap2 = q_biquad_from_z_rtheta(rp1, thp1)
    ap3, ap4 = q_biquad_from_z_rtheta(rp2, thp2)

    def poly2_mul(a1, a2, b1, b2):
        c0 = torch.ones_like(a1)
        c1 = a1 + b1
        c2 = a2 + a1 * b1 + b2
        c3 = a1 * b2 + a2 * b1
        c4 = a2 * b2
        return torch.stack([c0, c1, c2, c3, c4], dim=1)

    b = poly2_mul(bz1, bz2, bz3, bz4)
    a = poly2_mul(ap1, ap2, ap3, ap4)

    # normalize a0=1
    a0 = a[:, 0:1]
    b = b / (a0 + 1e-12)
    a = a / (a0 + 1e-12)

    # scale numerator so that b0 ~= g0
    b_scale = g0.unsqueeze(1) / (b[:, 0:1] + 1e-12)
    b = b * b_scale

    return b, a


# ============================================================
# Build dataset cache
# ============================================================

def build_cache(rows, normalizer: ParamNormalizer):
    params = np.stack([r["proto"] for r in rows], axis=0).astype(np.float32)
    params_n = normalizer.encode_np(params)

    structs = np.stack([extract_structured_repr(r) for r in rows], axis=0).astype(np.float32)

    b = np.stack([r["b"] for r in rows], axis=0).astype(np.float32)
    a = np.stack([r["a"] for r in rows], axis=0).astype(np.float32)
    err = np.array([r["err"] for r in rows], dtype=np.float32)

    return {
        "params": params,
        "params_n": params_n,
        "structs": structs,
        "b": b,
        "a": a,
        "err": err,
    }


# ============================================================
# Neighbor selection
# ============================================================

def build_neighbor_pool(cache):
    params_n = cache["params_n"]
    n = params_n.shape[0]
    pools = []

    for i in range(n):
        d2 = np.sum((params_n - params_n[i:i+1]) ** 2, axis=1)
        order = np.argsort(d2)

        # exclude self
        order = order[order != i]
        order = order[:CFG.neighbor_pool]
        pools.append(order.astype(np.int32))

    return pools


def sample_training_batch(cache, neighbor_pools, batch_size, device):
    n = cache["params"].shape[0]

    target_idx = np.random.randint(0, n, size=batch_size)

    base_idx = []
    for ti in target_idx:
        cand = neighbor_pools[ti]
        if len(cand) == 0:
            bi = ti
        else:
            bi = int(np.random.choice(cand))
        base_idx.append(bi)
    base_idx = np.array(base_idx, dtype=np.int32)

    batch = {
        "target_params": torch.tensor(cache["params"][target_idx], dtype=torch.float32, device=device),
        "target_params_n": torch.tensor(cache["params_n"][target_idx], dtype=torch.float32, device=device),
        "target_b": torch.tensor(cache["b"][target_idx], dtype=torch.float32, device=device),
        "target_a": torch.tensor(cache["a"][target_idx], dtype=torch.float32, device=device),
        "base_params": torch.tensor(cache["params"][base_idx], dtype=torch.float32, device=device),
        "base_params_n": torch.tensor(cache["params_n"][base_idx], dtype=torch.float32, device=device),
        "base_struct": torch.tensor(cache["structs"][base_idx], dtype=torch.float32, device=device),
    }
    return batch


# ============================================================
# Model
# ============================================================

class StructuredTransportNet(nn.Module):
    """
    Input:
      base_struct(9) + base_params_n(4) + target_params_n(4) + delta_params(4)
      => 21 dims
    Output:
      residual on structured rep:
      [drz1, drz2, dthz1, dthz2, drp1, drp2, dthp1, dthp2, dg0]
    """
    def __init__(self, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(21, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 9),
        )

    def forward(self, base_struct, base_params_n, target_params_n):
        delta = target_params_n - base_params_n
        x = torch.cat([base_struct, base_params_n, target_params_n, delta], dim=1)
        return self.net(x)


# ============================================================
# Structured transport
# ============================================================

def apply_transport(base_struct: torch.Tensor, raw_out: torch.Tensor):
    """
    base_struct: [B,9]
    raw_out: [B,9]
    """
    out = base_struct.clone()

    # base layout:
    # [rz1, rz2, thz1, thz2, rp1, rp2, thp1, thp2, g0]
    dr = CFG.dr_scale * torch.tanh(raw_out[:, [0, 1, 4, 5]])
    dth = CFG.dtheta_scale * torch.tanh(raw_out[:, [2, 3, 6, 7]])
    dg = CFG.dg_scale * torch.tanh(raw_out[:, [8]])

    # radii
    out[:, 0] = torch.clamp(base_struct[:, 0] + dr[:, 0], min=1e-3)
    out[:, 1] = torch.clamp(base_struct[:, 1] + dr[:, 1], min=1e-3)
    out[:, 4] = torch.clamp(base_struct[:, 4] + dr[:, 2], min=1e-3)
    out[:, 5] = torch.clamp(base_struct[:, 5] + dr[:, 3], min=1e-3)

    # angles in [0, pi]
    out[:, 2] = torch.clamp(base_struct[:, 2] + dth[:, 0], 1e-4, math.pi - 1e-4)
    out[:, 3] = torch.clamp(base_struct[:, 3] + dth[:, 1], 1e-4, math.pi - 1e-4)
    out[:, 6] = torch.clamp(base_struct[:, 6] + dth[:, 2], 1e-4, math.pi - 1e-4)
    out[:, 7] = torch.clamp(base_struct[:, 7] + dth[:, 3], 1e-4, math.pi - 1e-4)

    # keep ordering by angle
    thz = out[:, 2:4]
    idxz = torch.argsort(thz, dim=1)
    rz = out[:, 0:2].gather(1, idxz)
    thz = thz.gather(1, idxz)
    out[:, 0:2] = rz
    out[:, 2:4] = thz

    thp = out[:, 6:8]
    idxp = torch.argsort(thp, dim=1)
    rp = out[:, 4:6].gather(1, idxp)
    thp = thp.gather(1, idxp)
    out[:, 4:6] = rp
    out[:, 6:8] = thp

    # gain scalar
    out[:, 8:9] = torch.clamp(base_struct[:, 8:9] * torch.exp(dg), min=1e-5)

    return out


# ============================================================
# Losses
# ============================================================

def response_loss_db(pred_mag: torch.Tensor, tgt_mag: torch.Tensor):
    pred_db = db_torch(pred_mag)
    tgt_db = db_torch(tgt_mag)

    w = 0.08 + 0.92 / (1.0 + torch.exp(-(tgt_db + 40.0) / 4.0))
    return torch.mean(w * (pred_db - tgt_db) ** 2)


def move_loss(pred_struct: torch.Tensor, base_struct: torch.Tensor):
    return F.smooth_l1_loss(pred_struct, base_struct)


def smoothness_loss(model, base_struct, base_params_n, target_params, target_params_n, normalizer, device):
    # perturb target params slightly
    tp2 = target_params.clone()
    tp2[:, 0] = tp2[:, 0] * torch.exp(torch.randn_like(tp2[:, 0]) * CFG.delta_log_fc)
    tp2[:, 1] = tp2[:, 1] * torch.exp(torch.randn_like(tp2[:, 1]) * CFG.delta_log_q)
    tp2[:, 2] = tp2[:, 2] + torch.randn_like(tp2[:, 2]) * CFG.delta_gain
    tp2[:, 3] = tp2[:, 3] + torch.randn_like(tp2[:, 3]) * CFG.delta_stages
    tp2 = normalizer.clamp_torch(tp2)
    tp2_n = normalizer.encode_torch(tp2)

    raw1 = model(base_struct, base_params_n, target_params_n)
    raw2 = model(base_struct, base_params_n, tp2_n)

    ps1 = apply_transport(base_struct, raw1)
    ps2 = apply_transport(base_struct, raw2)

    dcoef = ps1 - ps2
    dpar = torch.norm(target_params_n - tp2_n, dim=1, keepdim=True) + 1e-6
    return torch.mean(torch.sum(dcoef ** 2, dim=1, keepdim=True) / dpar)


# ============================================================
# Save / load
# ============================================================

def save_ckpt(path, model, opt, epoch, best_loss, normalizer):
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "config": CFG.__dict__,
        "normalizer": {
            "fc_min": normalizer.fc_min,
            "fc_max": normalizer.fc_max,
            "q_min": normalizer.q_min,
            "q_max": normalizer.q_max,
            "g_min": normalizer.g_min,
            "g_max": normalizer.g_max,
            "s_min": normalizer.s_min,
            "s_max": normalizer.s_max,
        },
        "format": "structured_transport_v1",
    }
    torch.save(ckpt, path)


def maybe_load_ckpt(path, model, opt, device):
    if os.path.isfile(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        return int(ckpt.get("epoch", 0)), float(ckpt.get("best_loss", 1e30))
    return 0, 1e30


# ============================================================
# Training
# ============================================================

def train():
    set_seed(CFG.seed)

    os.chdir(CFG.src_dir)
    device = torch.device(CFG.device)

    rows = parse_lut_file(CFG.lut_file)
    normalizer = ParamNormalizer(rows)
    cache = build_cache(rows, normalizer)
    neighbor_pools = build_neighbor_pool(cache)

    w_grid_np, freqs_np = make_log_freq_grid(CFG.fs, CFG.f_min_resp, CFG.num_freq)
    w_grid = torch.tensor(w_grid_np, dtype=torch.float32, device=device)
    freqs = torch.tensor(freqs_np, dtype=torch.float32, device=device)

    model = StructuredTransportNet(hidden=CFG.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    start_epoch, best_loss = maybe_load_ckpt(CFG.save_file, model, opt, device)

    print(f"device        : {device}")
    print(f"lut rows       : {len(rows)}")
    print(f"save file      : {os.path.join(CFG.src_dir, CFG.save_file)}")
    if start_epoch > 0:
        print(f"resume epoch   : {start_epoch}")
        print(f"best_loss      : {best_loss:.6f}")
    print()

    epoch = start_epoch

    while True:
        epoch += 1
        model.train()

        s_total = 0.0
        s_proto = 0.0
        s_lut = 0.0
        s_move = 0.0
        s_smooth = 0.0

        for _ in range(CFG.steps_per_epoch):
            batch = sample_training_batch(cache, neighbor_pools, CFG.batch_size, device)

            base_struct = batch["base_struct"]
            base_params_n = batch["base_params_n"]
            target_params = batch["target_params"]
            target_params_n = batch["target_params_n"]
            target_b = batch["target_b"]
            target_a = batch["target_a"]

            raw = model(base_struct, base_params_n, target_params_n)
            pred_struct = apply_transport(base_struct, raw)
            pred_b, pred_a = reconstruct_from_struct_torch(pred_struct)

            pred_mag = iir_mag_response_torch(pred_b, pred_a, w_grid)
            tgt_proto_mag = prototype_mag_response_torch(target_params, freqs)
            tgt_lut_mag = iir_mag_response_torch(target_b, target_a, w_grid)

            loss_proto = response_loss_db(pred_mag, tgt_proto_mag)
            loss_lut = response_loss_db(pred_mag, tgt_lut_mag)
            loss_move = move_loss(pred_struct, base_struct)
            loss_smooth = smoothness_loss(
                model, base_struct, base_params_n, target_params, target_params_n, normalizer, device
            )

            loss = (
                CFG.w_proto * loss_proto +
                CFG.w_lut_resp * loss_lut +
                CFG.w_move * loss_move +
                CFG.w_smooth * loss_smooth
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            s_total += float(loss.item())
            s_proto += float(loss_proto.item())
            s_lut += float(loss_lut.item())
            s_move += float(loss_move.item())
            s_smooth += float(loss_smooth.item())

        avg_total = s_total / CFG.steps_per_epoch
        avg_proto = s_proto / CFG.steps_per_epoch
        avg_lut = s_lut / CFG.steps_per_epoch
        avg_move = s_move / CFG.steps_per_epoch
        avg_smooth = s_smooth / CFG.steps_per_epoch

        if epoch % CFG.print_every == 0:
            print(
                f"epoch {epoch:6d} | "
                f"loss={avg_total:.6f} | "
                f"proto={avg_proto:.6f} | "
                f"lut={avg_lut:.6f} | "
                f"move={avg_move:.6f} | "
                f"smooth={avg_smooth:.6f}"
            )

        if avg_total < best_loss:
            best_loss = avg_total
            save_ckpt(CFG.save_file, model, opt, epoch, best_loss, normalizer)
            print(f"  saved new best | best_loss={best_loss:.6f}")
        elif epoch % CFG.save_every == 0:
            save_ckpt(CFG.save_file, model, opt, epoch, best_loss, normalizer)


if __name__ == "__main__":
    train()