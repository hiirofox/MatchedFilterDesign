import os
import math
import json
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Source directory
# ============================================================
SRC_DIR = r"D:\Projects\Py\matchedfilterdesign\iirCoeffsLuts"
os.chdir(SRC_DIR)

LUT_FILE = os.path.join(SRC_DIR, "iirluts2b.txt")
BEST_FILE = os.path.join(SRC_DIR, "best.pt")
STATE_JSON = os.path.join(SRC_DIR, "best_state.json")


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    fs: float = 48000.0
    num_freq: int = 768
    f_min_resp: float = 70.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1234

    # very small model
    hidden: int = 32

    # batch
    batch_lut: int = 384
    batch_cont: int = 384

    # optimizer
    lr: float = 1e-3
    weight_decay: float = 0.0

    # parameter ranges
    fc_min: float = 20.0
    fc_max: float = 24000.0
    q_min: float = 0.5
    q_max: float = 20.0
    gain_min: float = -18.0
    gain_max: float = 18.0
    stages_min: float = 1.125
    stages_max: float = 3.0

    # training weights
    w_lut_resp: float = 2.0
    w_cont_resp: float = 1.0
    w_smooth: float = 0.01

    # training behavior
    steps_per_epoch: int = 300
    print_every_epoch: int = 1
    save_every_epoch: int = 1

    # light smoothing perturbation
    delta_log_fc: float = 0.020
    delta_log_q: float = 0.020
    delta_gain: float = 0.40
    delta_stages: float = 0.05

    # pole radius limit
    pole_r_max: float = 0.9992

    # zero radius limit
    zero_r_max: float = 1.4

    # gain scaling
    gain_scale_exp: float = 0.7


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


def mag_to_db(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return 20.0 * torch.log10(x + eps)


# ============================================================
# LUT parsing
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
            proto = np.array(vals[4:8], dtype=np.float32)
            b = np.array(vals[9:14], dtype=np.float32)
            a = np.array(vals[14:19], dtype=np.float32)

            rows.append({
                "proto": proto,
                "b": b,
                "a": a,
            })

    if not rows:
        raise RuntimeError(f"No valid rows found in {path}")
    return rows


# ============================================================
# Normalizer
# ============================================================
class ParamNormalizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log_fc_min = math.log(cfg.fc_min)
        self.log_fc_max = math.log(cfg.fc_max)
        self.log_q_min = math.log(cfg.q_min)
        self.log_q_max = math.log(cfg.q_max)

    def encode_torch(self, x: torch.Tensor) -> torch.Tensor:
        fc, q, g, s = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        x0 = (torch.log(fc) - self.log_fc_min) / (self.log_fc_max - self.log_fc_min) * 2.0 - 1.0
        x1 = (torch.log(q) - self.log_q_min) / (self.log_q_max - self.log_q_min) * 2.0 - 1.0
        x2 = (g - self.cfg.gain_min) / (self.cfg.gain_max - self.cfg.gain_min) * 2.0 - 1.0
        x3 = (s - self.cfg.stages_min) / (self.cfg.stages_max - self.cfg.stages_min) * 2.0 - 1.0
        return torch.stack([x0, x1, x2, x3], dim=1)

    def clamp_torch(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        y[:, 0] = y[:, 0].clamp(self.cfg.fc_min, self.cfg.fc_max)
        y[:, 1] = y[:, 1].clamp(self.cfg.q_min, self.cfg.q_max)
        y[:, 2] = y[:, 2].clamp(self.cfg.gain_min, self.cfg.gain_max)
        y[:, 3] = y[:, 3].clamp(self.cfg.stages_min, self.cfg.stages_max)
        return y


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
def iir_mag_response_torch(b: torch.Tensor, a: torch.Tensor, w_grid: torch.Tensor) -> torch.Tensor:
    k = torch.arange(5, device=b.device, dtype=w_grid.dtype)
    ejwk = torch.exp(-1j * w_grid[:, None] * k[None, :])

    num = torch.sum(b[:, None, :] * ejwk[None, :, :], dim=-1)
    den = torch.sum(a[:, None, :] * ejwk[None, :, :], dim=-1)
    H = num / (den + 1e-12)
    return torch.abs(H)


# ============================================================
# Pole-zero parameterization
# ============================================================
def poly2_mul(a1, a2, b1, b2):
    c0 = torch.ones_like(a1)
    c1 = a1 + b1
    c2 = a2 + a1 * b1 + b2
    c3 = a1 * b2 + a2 * b1
    c4 = a2 * b2
    return torch.stack([c0, c1, c2, c3, c4], dim=1)


def biquad_from_r_theta(r, theta):
    c1 = -2.0 * r * torch.cos(theta)
    c2 = r * r
    return c1, c2


def coeffs_from_raw(raw: torch.Tensor, cfg: Config):
    zr1_raw, zt1_raw, zr2_raw, zt2_raw, pr1_raw, pt1_raw, pr2_raw, pt2_raw, g_raw = torch.chunk(raw, 9, dim=1)

    zr1_raw = zr1_raw.squeeze(1)
    zt1_raw = zt1_raw.squeeze(1)
    zr2_raw = zr2_raw.squeeze(1)
    zt2_raw = zt2_raw.squeeze(1)
    pr1_raw = pr1_raw.squeeze(1)
    pt1_raw = pt1_raw.squeeze(1)
    pr2_raw = pr2_raw.squeeze(1)
    pt2_raw = pt2_raw.squeeze(1)
    g_raw = g_raw.squeeze(1)

    # zeros can be outside unit circle
    rz1 = cfg.zero_r_max * torch.sigmoid(zr1_raw)
    rz2 = cfg.zero_r_max * torch.sigmoid(zr2_raw)

    # poles forced inside unit circle with smooth sigmoid
    rp1 = 1e-4 + (cfg.pole_r_max - 1e-4) * torch.sigmoid(pr1_raw)
    rp2 = 1e-4 + (cfg.pole_r_max - 1e-4) * torch.sigmoid(pr2_raw)

    thz1 = math.pi * torch.sigmoid(zt1_raw)
    thz2 = math.pi * torch.sigmoid(zt2_raw)
    thp1 = math.pi * torch.sigmoid(pt1_raw)
    thp2 = math.pi * torch.sigmoid(pt2_raw)

    bz1, bz2 = biquad_from_r_theta(rz1, thz1)
    bz3, bz4 = biquad_from_r_theta(rz2, thz2)
    az1, az2 = biquad_from_r_theta(rp1, thp1)
    az3, az4 = biquad_from_r_theta(rp2, thp2)

    b = poly2_mul(bz1, bz2, bz3, bz4)
    a = poly2_mul(az1, az2, az3, az4)

    g = torch.exp(cfg.gain_scale_exp * g_raw).unsqueeze(1)
    b = b * g

    return b, a


# ============================================================
# Tiny model
# ============================================================
class TinyPZIIRNet(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 9),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Sampling
# ============================================================
def sample_lut_batch(rows, batch_size: int, device):
    idx = np.random.randint(0, len(rows), size=batch_size)
    params = np.stack([rows[i]["proto"] for i in idx], axis=0).astype(np.float32)
    return torch.tensor(params, dtype=torch.float32, device=device)


def sample_continuous_params(batch_size: int, cfg: Config, device):
    u0 = torch.rand(batch_size, device=device)
    u1 = torch.rand(batch_size, device=device)
    u2 = torch.rand(batch_size, device=device)
    u3 = torch.rand(batch_size, device=device)

    fc = torch.exp(math.log(cfg.fc_min) + u0 * (math.log(cfg.fc_max) - math.log(cfg.fc_min)))
    q = torch.exp(math.log(cfg.q_min) + u1 * (math.log(cfg.q_max) - math.log(cfg.q_min)))
    g = cfg.gain_min + u2 * (cfg.gain_max - cfg.gain_min)
    s = cfg.stages_min + u3 * (cfg.stages_max - cfg.stages_min)

    return torch.stack([fc, q, g, s], dim=1)


def perturb_params(params: torch.Tensor, cfg: Config, normalizer: ParamNormalizer):
    x = params.clone()
    x[:, 0] = x[:, 0] * torch.exp(torch.randn_like(x[:, 0]) * cfg.delta_log_fc)
    x[:, 1] = x[:, 1] * torch.exp(torch.randn_like(x[:, 1]) * cfg.delta_log_q)
    x[:, 2] = x[:, 2] + torch.randn_like(x[:, 2]) * cfg.delta_gain
    x[:, 3] = x[:, 3] + torch.randn_like(x[:, 3]) * cfg.delta_stages
    return normalizer.clamp_torch(x)


# ============================================================
# Loss
# ============================================================
def response_loss_db(pred_mag: torch.Tensor, tgt_mag: torch.Tensor) -> torch.Tensor:
    pred_db = mag_to_db(pred_mag)
    tgt_db = mag_to_db(tgt_mag)

    w = 0.08 + 0.92 / (1.0 + torch.exp(-(tgt_db + 40.0) / 4.0))
    return torch.mean(w * (pred_db - tgt_db) ** 2)


def smoothness_loss(raw1: torch.Tensor, raw2: torch.Tensor, p1n: torch.Tensor, p2n: torch.Tensor) -> torch.Tensor:
    d_raw = raw1 - raw2
    d_par = torch.norm(p1n - p2n, dim=1, keepdim=True) + 1e-6
    return torch.mean(torch.sum(d_raw ** 2, dim=1, keepdim=True) / d_par)


# ============================================================
# Save / Load
# ============================================================
def save_checkpoint(model, optimizer, epoch: int, best_loss: float, cfg: Config):
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "config": asdict(cfg),
        "format": "pz_iir_v1",
    }
    torch.save(ckpt, BEST_FILE)

    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "epoch": epoch,
            "best_loss": best_loss,
            "best_file": BEST_FILE,
            "format": "pz_iir_v1",
        }, f, indent=2)


def load_checkpoint_if_exists(model, optimizer, device):
    start_epoch = 0
    best_loss = float("inf")

    if os.path.isfile(BEST_FILE):
        print(f"Found existing checkpoint: {BEST_FILE}")
        ckpt = torch.load(BEST_FILE, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_loss = float(ckpt.get("best_loss", float("inf")))
        print(f"Resume from epoch {start_epoch}, best_loss={best_loss:.6f}")
    else:
        print("No existing best.pt found. Start from scratch.")

    return start_epoch, best_loss


# ============================================================
# Training
# ============================================================
def train():
    set_seed(CFG.seed)
    device = torch.device(CFG.device)
    normalizer = ParamNormalizer(CFG)

    rows = parse_lut_file(LUT_FILE)
    w_grid_np, freqs_np = make_log_freq_grid(CFG.fs, CFG.f_min_resp, CFG.num_freq)
    w_grid = torch.tensor(w_grid_np, dtype=torch.float32, device=device)
    freqs = torch.tensor(freqs_np, dtype=torch.float32, device=device)

    model = TinyPZIIRNet(hidden=CFG.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    start_epoch, best_loss = load_checkpoint_if_exists(model, optimizer, device)

    print(f"Device: {device}")
    print(f"LUT rows: {len(rows)}")
    print(f"Training file: {LUT_FILE}")
    print(f"Output weight: {BEST_FILE}")
    print("Training will continue indefinitely until you stop it.\n")

    epoch = start_epoch

    while True:
        epoch += 1

        model.train()
        loss_sum = 0.0
        lut_sum = 0.0
        cont_sum = 0.0
        smooth_sum = 0.0

        # curriculum: earlier epochs emphasize LUT points more
        if epoch <= 50:
            lut_mul = 1.3
            cont_mul = 0.7
        elif epoch <= 150:
            lut_mul = 1.0
            cont_mul = 1.0
        else:
            lut_mul = 0.9
            cont_mul = 1.1

        for _ in range(CFG.steps_per_epoch):
            # LUT batch
            params_lut = sample_lut_batch(rows, CFG.batch_lut, device)
            params_lut_n = normalizer.encode_torch(params_lut)
            raw_lut = model(params_lut_n)
            b_lut, a_lut = coeffs_from_raw(raw_lut, CFG)

            pred_mag_lut = iir_mag_response_torch(b_lut, a_lut, w_grid)
            tgt_mag_lut = prototype_mag_response_torch(params_lut, freqs)
            loss_lut = response_loss_db(pred_mag_lut, tgt_mag_lut)

            # Continuous batch
            params_cont = sample_continuous_params(CFG.batch_cont, CFG, device)
            params_cont_n = normalizer.encode_torch(params_cont)
            raw_cont = model(params_cont_n)
            b_cont, a_cont = coeffs_from_raw(raw_cont, CFG)

            pred_mag_cont = iir_mag_response_torch(b_cont, a_cont, w_grid)
            tgt_mag_cont = prototype_mag_response_torch(params_cont, freqs)
            loss_cont = response_loss_db(pred_mag_cont, tgt_mag_cont)

            # Very light smoothness on raw latent outputs
            params_cont_2 = perturb_params(params_cont, CFG, normalizer)
            params_cont_2_n = normalizer.encode_torch(params_cont_2)
            raw_cont_2 = model(params_cont_2_n)

            loss_smooth = smoothness_loss(raw_cont, raw_cont_2, params_cont_n, params_cont_2_n)

            loss = (
                lut_mul * CFG.w_lut_resp * loss_lut +
                cont_mul * CFG.w_cont_resp * loss_cont +
                CFG.w_smooth * loss_smooth
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            lut_sum += float(loss_lut.item())
            cont_sum += float(loss_cont.item())
            smooth_sum += float(loss_smooth.item())

        loss_avg = loss_sum / CFG.steps_per_epoch
        lut_avg = lut_sum / CFG.steps_per_epoch
        cont_avg = cont_sum / CFG.steps_per_epoch
        smooth_avg = smooth_sum / CFG.steps_per_epoch

        if epoch % CFG.print_every_epoch == 0:
            print(
                f"Epoch {epoch:7d} | "
                f"loss={loss_avg:.6f} | "
                f"lut={lut_avg:.6f} | "
                f"cont={cont_avg:.6f} | "
                f"smooth={smooth_avg:.6f}"
            )

        if loss_avg < best_loss:
            best_loss = loss_avg
            save_checkpoint(model, optimizer, epoch, best_loss, CFG)
            print(f"  saved improved best.pt | best_loss={best_loss:.6f}")
        elif epoch % CFG.save_every_epoch == 0:
            save_checkpoint(model, optimizer, epoch, best_loss, CFG)


if __name__ == "__main__":
    train()