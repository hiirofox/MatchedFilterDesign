import math
import os
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    sample_rate: float = 48000.0
    n_fft: int = 8192

    # User-specified total IIR order (must be even)
    order: int = 4

    hidden_dim: int = 96
    depth: int = 6
    batch_size: int = 128
    num_steps: int = 3000
    lr: float = 2e-3
    device: str = "cuda" 

    # saved files
    checkpoint_path: str = "rootnet_iir_checkpoint.pt"
    weights_path: str = "rootnet_iir_weights.pt"

    # parameter ranges
    f0_min: float = 20.0
    f0_max: float = 24000.0
    q_min: float = 0.1
    q_max: float = 24.0
    gain_min_db: float = -30.0
    gain_max_db: float = 30.0

    # root constraints
    pole_r_max: float = 1.0
    zero_r_max: float = 1.0   # <= 1 for minimum-phase zeros

    # loss weights
    w_mag: float = 1.0
    w_cplx: float = 0#0.08
    w_smooth: float = 0#0.02
    w_margin: float = 0#0.002    
    w_anchor: float = 0.2#0.8

    # smoothing deltas
    delta_f0_rel: float = 0#0.01
    delta_q_rel: float = 0#0.03
    delta_gain_abs: float = 0#0.25

    seed: int = 114514

    @property
    def n_sections(self):
        assert self.order % 2 == 0, "order must be even."
        return self.order // 2


CFG = Config()
torch.manual_seed(CFG.seed)


# ============================================================
# Utility
# ============================================================

def db20(x, eps=1e-8):
    return 20.0 * torch.log10(torch.clamp(x, min=eps))

def normalize_params(params, cfg: Config):
    """
    params: (..., 3) = [f0, Q, gain_db]
    """
    f0 = params[..., 0]
    q = params[..., 1]
    g = params[..., 2]

    f0_min_t = torch.tensor(cfg.f0_min, device=f0.device, dtype=f0.dtype)
    f0_max_t = torch.tensor(cfg.f0_max, device=f0.device, dtype=f0.dtype)

    f0_n = (torch.log2(f0) - torch.log2(f0_min_t)) / (torch.log2(f0_max_t) - torch.log2(f0_min_t))
    q_n = (torch.log(q) - math.log(cfg.q_min)) / (math.log(cfg.q_max) - math.log(cfg.q_min))
    g_n = (g - cfg.gain_min_db) / (cfg.gain_max_db - cfg.gain_min_db)

    x = torch.stack([f0_n, q_n, g_n], dim=-1)
    return 2.0 * x - 1.0

def sample_params(batch_size, cfg: Config, device):
    u1 = torch.rand(batch_size, device=device)
    u2 = torch.rand(batch_size, device=device)
    u3 = torch.rand(batch_size, device=device)

    f0 = cfg.f0_min * (cfg.f0_max / cfg.f0_min) ** u1
    q = cfg.q_min * (cfg.q_max / cfg.q_min) ** u2
    gain_db = cfg.gain_min_db + (cfg.gain_max_db - cfg.gain_min_db) * u3
    return torch.stack([f0, q, gain_db], dim=-1)

def perturb_params(params, cfg: Config):
    f0, q, g = params[..., 0], params[..., 1], params[..., 2]

    df = (torch.rand_like(f0) * 2 - 1) * cfg.delta_f0_rel
    dq = (torch.rand_like(q) * 2 - 1) * cfg.delta_q_rel
    dg = (torch.rand_like(g) * 2 - 1) * cfg.delta_gain_abs

    f0_2 = torch.clamp(f0 * (1.0 + df), cfg.f0_min, cfg.f0_max)
    q_2 = torch.clamp(q * (1.0 + dq), cfg.q_min, cfg.q_max)
    g_2 = torch.clamp(g + dg, cfg.gain_min_db, cfg.gain_max_db)

    return torch.stack([f0_2, q_2, g_2], dim=-1)

def gather_lastdim(x, idx):
    return torch.gather(x, dim=-1, index=idx)


# ============================================================
# Analog peaking prototype
# ============================================================
# Analog bell/peaking prototype:
#
#               s^2 + (A/Q) * w0 * s + w0^2
# H(s) = ------------------------------------------------
#         s^2 + (1/(A Q)) * w0 * s + w0^2
#
# where A = 10^(gain_db/40)
#
# This is an analog peaking prototype with:
# - unity gain at DC and infinity
# - boost/cut around w0
# - symmetric boost/cut behavior
#
# We evaluate it on s = j * 2*pi*f
# ============================================================

def analog_peaking_response(params, freqs_hz):
    """
    params: [B, 3] = [f0, Q, gain_db]
    freqs_hz: [F]
    return: H_target [B, F] complex
    """
    device = params.device
    dtype = params.dtype

    f0 = params[:, 0:1]         # [B,1]
    Q = params[:, 1:2]
    gain_db = params[:, 2:3]

    A = torch.pow(torch.tensor(10.0, device=device, dtype=dtype), gain_db / 40.0)
    w0 = 2.0 * math.pi * f0     # [B,1]

    Omega = 2.0 * math.pi * freqs_hz[None, :]   # [1,F]
    s = 1j * Omega                              # [1,F]

    num = s**2 + (A / Q) * w0 * s + w0**2
    den = s**2 + (1.0 / (A * Q)) * w0 * s + w0**2
    H = num / den
    return H


# ============================================================
# Network
# ============================================================

class RootNet(nn.Module):
    def __init__(self, order=4, hidden_dim=96, depth=3):
        super().__init__()
        assert order % 2 == 0, "order must be even."
        self.order = order
        self.n_sections = order // 2

        layers = []
        in_dim = 3
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        self.backbone = nn.Sequential(*layers)

        # [pole_r, pole_theta, zero_r, zero_theta] * n_sections + 1 global gain
        self.head = nn.Linear(hidden_dim, 4 * self.n_sections + 1)

    def forward(self, x):
        h = self.backbone(x)
        out = self.head(h)

        ns = self.n_sections
        pole_r_raw = out[:, 0:ns]
        pole_t_raw = out[:, ns:2*ns]
        zero_r_raw = out[:, 2*ns:3*ns]
        zero_t_raw = out[:, 3*ns:4*ns]
        log_gain_raw = out[:, 4*ns:4*ns+1]

        return {
            "pole_r_raw": pole_r_raw,
            "pole_t_raw": pole_t_raw,
            "zero_r_raw": zero_r_raw,
            "zero_t_raw": zero_t_raw,
            "log_gain_raw": log_gain_raw,
        }


# ============================================================
# Root parameterization and DSP layer
# ============================================================

class RootParameterization(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def map_roots(self, net_out):
        pole_r = self.cfg.pole_r_max * torch.sigmoid(net_out["pole_r_raw"])
        zero_r = self.cfg.zero_r_max * torch.sigmoid(net_out["zero_r_raw"])
        # [0, pi]
        pole_theta = math.pi * torch.sigmoid(net_out["pole_t_raw"])
        zero_theta = math.pi * torch.sigmoid(net_out["zero_t_raw"])

        # sort by pole angle to reduce section permutation
        idx = torch.argsort(pole_theta, dim=-1)
        pole_theta = gather_lastdim(pole_theta, idx)
        pole_r = gather_lastdim(pole_r, idx)
        zero_theta = gather_lastdim(zero_theta, idx)
        zero_r = gather_lastdim(zero_r, idx)

        global_gain = torch.exp(net_out["log_gain_raw"])
        return pole_r, pole_theta, zero_r, zero_theta, global_gain

    def section_coeffs(self, pole_r, pole_theta, zero_r, zero_theta):
        """
        Each conjugate pair maps to a real biquad:
        B(z) = 1 - 2*r_z*cos(theta_z) z^-1 + r_z^2 z^-2
        A(z) = 1 - 2*r_p*cos(theta_p) z^-1 + r_p^2 z^-2
        """
        cz = torch.cos(zero_theta)
        cp = torch.cos(pole_theta)

        b0 = torch.ones_like(zero_r)
        b1 = -2.0 * zero_r * cz
        b2 = zero_r ** 2

        a0 = torch.ones_like(pole_r)
        a1 = -2.0 * pole_r * cp
        a2 = pole_r ** 2

        b = torch.stack([b0, b1, b2], dim=-1)
        a = torch.stack([a0, a1, a2], dim=-1)
        return b, a

    def freq_response(self, pole_r, pole_theta, zero_r, zero_theta, global_gain, omega):
        """
        omega: [F] on [0, pi]
        return: H [B, F] complex
        """
        b, a = self.section_coeffs(pole_r, pole_theta, zero_r, zero_theta)

        z1 = torch.exp(-1j * omega)[None, None, :]   # [1,1,F]
        z2 = torch.exp(-2j * omega)[None, None, :]

        num = b[..., 0:1] + b[..., 1:2] * z1 + b[..., 2:3] * z2
        den = a[..., 0:1] + a[..., 1:2] * z1 + a[..., 2:3] * z2

        H_sec = num / den
        H = torch.prod(H_sec, dim=1)
        H = global_gain * H
        return H

    def roots_to_sos(self, pole_r, pole_theta, zero_r, zero_theta, global_gain):
        """
        Return SOS [B, ns, 6] = [b0,b1,b2,a0,a1,a2]
        Global gain is folded into the first section numerator.
        """
        b, a = self.section_coeffs(pole_r, pole_theta, zero_r, zero_theta)
        b = b.clone()
        b[:, 0, :] *= global_gain
        sos = torch.cat([b, a], dim=-1)
        return sos


# ============================================================
# Loss
# ============================================================

def compute_losses(model, root_layer, params, omega, freqs_hz, cfg: Config):
    x = normalize_params(params, cfg)
    net_out = model(x)

    pole_r, pole_theta, zero_r, zero_theta, global_gain = root_layer.map_roots(net_out)

    H_pred = root_layer.freq_response(pole_r, pole_theta, zero_r, zero_theta, global_gain, omega)
    H_tgt = analog_peaking_response(params, freqs_hz)

    mag_pred_db = db20(torch.abs(H_pred))
    mag_tgt_db = db20(torch.abs(H_tgt))
    loss_mag = F.mse_loss(mag_pred_db, mag_tgt_db)

    loss_cplx = (
        F.mse_loss(torch.real(H_pred), torch.real(H_tgt)) +
        F.mse_loss(torch.imag(H_pred), torch.imag(H_tgt))
    )

    loss_margin = torch.mean(F.relu(pole_r - 0.95) ** 2)

    # smoothness regularization
    params2 = perturb_params(params, cfg)
    x2 = normalize_params(params2, cfg)
    net_out2 = model(x2)
    pole_r2, pole_theta2, zero_r2, zero_theta2, global_gain2 = root_layer.map_roots(net_out2)

    dp = (normalize_params(params2, cfg) - normalize_params(params, cfg)).abs().mean(dim=-1, keepdim=True) + 1e-6

    smooth_roots = (
        (pole_r - pole_r2).pow(2).mean(dim=-1, keepdim=True) +
        (pole_theta - pole_theta2).pow(2).mean(dim=-1, keepdim=True) +
        (zero_r - zero_r2).pow(2).mean(dim=-1, keepdim=True) +
        (zero_theta - zero_theta2).pow(2).mean(dim=-1, keepdim=True) +
        (torch.log(global_gain + 1e-8) - torch.log(global_gain2 + 1e-8)).pow(2)
    ) / dp
    loss_smooth = smooth_roots.mean()

    # ------------------------------------------------------------
    # Anchor-gain loss:
    # enforce gain match at f0 and at 0.98 * Nyquist
    # ------------------------------------------------------------
    f0_anchor = params[:, 0]                                  # [B]
    fnyq_anchor = torch.full_like(f0_anchor, 0.9 * 0.5 * cfg.sample_rate)

    anchor_freqs = torch.stack([f0_anchor, fnyq_anchor], dim=-1)   # [B, 2]
    anchor_omega = 2.0 * math.pi * anchor_freqs / cfg.sample_rate  # [B, 2]

    # predicted response at anchor frequencies
    z1 = torch.exp(-1j * anchor_omega)[:, None, :]   # [B,1,2]
    z2 = torch.exp(-2j * anchor_omega)[:, None, :]   # [B,1,2]

    b, a = root_layer.section_coeffs(pole_r, pole_theta, zero_r, zero_theta)  # [B,ns,3]

    num_a = b[..., 0:1] + b[..., 1:2] * z1 + b[..., 2:3] * z2   # [B,ns,2]
    den_a = a[..., 0:1] + a[..., 1:2] * z1 + a[..., 2:3] * z2   # [B,ns,2]
    H_pred_anchor = torch.prod(num_a / den_a, dim=1)            # [B,2]
    H_pred_anchor = global_gain * H_pred_anchor                  # [B,2]

    # target response at the same anchor frequencies
    H_tgt_anchor = analog_peaking_response(params, anchor_freqs) # [B,2]

    loss_anchor = F.mse_loss(
        db20(torch.abs(H_pred_anchor)),
        db20(torch.abs(H_tgt_anchor))
    )

    loss = (
        cfg.w_mag * loss_mag +
        cfg.w_cplx * loss_cplx +
        cfg.w_smooth * loss_smooth +
        cfg.w_margin * loss_margin +
        cfg.w_anchor * loss_anchor
    )

    aux = {
        "loss_mag": loss_mag.detach(),
        "loss_cplx": loss_cplx.detach(),
        "loss_smooth": loss_smooth.detach(),
        "loss_margin": loss_margin.detach(),
        "loss_anchor": loss_anchor.detach(),
        "pole_r_mean": pole_r.mean().detach(),
    }
    return loss, aux


# ============================================================
# Save / load
# ============================================================

def save_model(model, cfg: Config, best_loss=None):
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": asdict(cfg),
        "best_loss": best_loss,
    }
    torch.save(checkpoint, cfg.checkpoint_path)
    torch.save(model.state_dict(), cfg.weights_path)

def load_model(checkpoint_path, device=None):
    payload = torch.load(checkpoint_path, map_location=device or "cpu")
    cfg = Config(**payload["config"])
    model = RootNet(order=cfg.order, hidden_dim=cfg.hidden_dim, depth=cfg.depth)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg


# ============================================================
# Train
# ============================================================

def train(cfg: Config):
    device = cfg.device

    assert cfg.order % 2 == 0, "order must be even."
    
    #linspace
    #omega = torch.linspace(0.0, math.pi, cfg.n_fft, device=device)
    #freqs_hz = omega * cfg.sample_rate / (2.0 * math.pi)
    # log-frequency grid, uniform in log(f)
    #logspace
    f_min = cfg.f0_min
    f_max = cfg.sample_rate * 0.5
    freqs_hz = torch.logspace(
        math.log10(f_min),
        math.log10(f_max),
        cfg.n_fft,
        device=device
    )
    omega = 2.0 * math.pi * freqs_hz / cfg.sample_rate

    model = RootNet(order=cfg.order, hidden_dim=cfg.hidden_dim, depth=cfg.depth).to(device)
    root_layer = RootParameterization(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_steps)

    best_loss = float("inf")

    print(f"Training RootNet with order={cfg.order}, n_sections={cfg.n_sections}")

    for step in range(1, cfg.num_steps + 1):
        model.train()
        params = sample_params(cfg.batch_size, cfg, device)

        optimizer.zero_grad()
        loss, aux = compute_losses(model, root_layer, params, omega, freqs_hz, cfg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        if step % 200 == 0 or step == 1:
            print(
                f"[{step:05d}/{cfg.num_steps}] "
                f"loss={loss.item():.6f} "
                f"mag={aux['loss_mag'].item():.6f} "
                f"cplx={aux['loss_cplx'].item():.6f} "
                f"smooth={aux['loss_smooth'].item():.6f} "
                f"margin={aux['loss_margin'].item():.6f} "
                f"anchor={aux['loss_anchor'].item():.6f} "
                f"pole_r_mean={aux['pole_r_mean'].item():.4f}"
            )

        if loss.item() < best_loss:
            best_loss = loss.item()
            save_model(model, cfg, best_loss=best_loss)

    # final save
    save_model(model, cfg, best_loss=best_loss)
    print(f"Training complete. Best loss = {best_loss:.6f}")
    print(f"Saved checkpoint to: {cfg.checkpoint_path}")
    print(f"Saved weights to:    {cfg.weights_path}")

    return model


# ============================================================
# Inference helpers
# ============================================================

@torch.no_grad()
def infer_roots_and_sos(model, cfg: Config, f0, Q, gain_db, device=None):
    device = device or cfg.device
    model = model.to(device).eval()
    root_layer = RootParameterization(cfg).to(device)

    params = torch.tensor([[f0, Q, gain_db]], dtype=torch.float32, device=device)
    x = normalize_params(params, cfg)
    out = model(x)

    pole_r, pole_theta, zero_r, zero_theta, global_gain = root_layer.map_roots(out)
    sos = root_layer.roots_to_sos(pole_r, pole_theta, zero_r, zero_theta, global_gain)

    return {
        "pole_r": pole_r[0].cpu(),
        "pole_theta": pole_theta[0].cpu(),
        "zero_r": zero_r[0].cpu(),
        "zero_theta": zero_theta[0].cpu(),
        "global_gain": global_gain[0, 0].cpu(),
        "sos": sos[0].cpu(),
    }

@torch.no_grad()
def infer_response(model, cfg: Config, f0, Q, gain_db, device=None):
    device = device or cfg.device
    model = model.to(device).eval()
    root_layer = RootParameterization(cfg).to(device)

    #omega = torch.linspace(0.0, math.pi, cfg.n_fft, device=device)
    #freqs_hz = omega * cfg.sample_rate / (2.0 * math.pi)
    # use the same log-frequency grid as training
    f_min = cfg.f0_min
    f_max = cfg.sample_rate * 0.5
    freqs_hz = torch.logspace(
        math.log10(f_min),
        math.log10(f_max),
        cfg.n_fft,
        device=device
    )
    omega = 2.0 * math.pi * freqs_hz / cfg.sample_rate

    params = torch.tensor([[f0, Q, gain_db]], dtype=torch.float32, device=device)
    x = normalize_params(params, cfg)
    out = model(x)
    pole_r, pole_theta, zero_r, zero_theta, global_gain = root_layer.map_roots(out)

    H_pred = root_layer.freq_response(pole_r, pole_theta, zero_r, zero_theta, global_gain, omega)[0]
    H_tgt = analog_peaking_response(params, freqs_hz)[0]

    return (
        freqs_hz.cpu().numpy(),
        db20(torch.abs(H_pred)).cpu().numpy(),
        db20(torch.abs(H_tgt)).cpu().numpy(),
    )


# ============================================================
# Interactive demo
# ============================================================

def interactive_demo(model, cfg: Config):
    f0_init = 1000.0
    q_init = 1.0
    g_init = 6.0

    freqs_hz, pred_db, tgt_db = infer_response(model, cfg, f0_init, q_init, g_init, cfg.device)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.28)

    ax.set_title(f"Analog Peaking Target vs Learned Digital IIR (order={cfg.order})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xscale("log")
    ax.set_xlim([20, cfg.sample_rate / 2])
    ax.set_ylim([-24, 24])
    ax.grid(True, which="both", ls="--", alpha=0.4)

    line_tgt, = ax.plot(freqs_hz, tgt_db, label="Analog target", linewidth=2)
    line_pred, = ax.plot(freqs_hz, pred_db, label="Model digital IIR", linewidth=2)
    ax.legend()

    ax_f0 = plt.axes([0.12, 0.16, 0.76, 0.03])
    ax_q  = plt.axes([0.12, 0.11, 0.76, 0.03])
    ax_g  = plt.axes([0.12, 0.06, 0.76, 0.03])

    slider_f0 = Slider(ax_f0, "f0", cfg.f0_min, cfg.f0_max, valinit=f0_init, valstep=1.0)
    slider_q  = Slider(ax_q,  "Q", cfg.q_min, cfg.q_max, valinit=q_init)
    slider_g  = Slider(ax_g,  "gain(dB)", cfg.gain_min_db, cfg.gain_max_db, valinit=g_init)

    def update(_):
        f0 = slider_f0.val
        Q = slider_q.val
        g = slider_g.val

        freqs_hz, pred_db, tgt_db = infer_response(model, cfg, f0, Q, g, cfg.device)
        line_tgt.set_ydata(tgt_db)
        line_pred.set_ydata(pred_db)
        fig.canvas.draw_idle()

        info = infer_roots_and_sos(model, cfg, f0, Q, g, cfg.device)
        print("\n=== Current Parameters ===")
        print(f"f0={f0:.2f} Hz, Q={Q:.4f}, gain={g:.2f} dB")
        print("Pole radii:", info["pole_r"].numpy())
        print("Pole theta:", info["pole_theta"].numpy())
        print("Zero radii:", info["zero_r"].numpy())
        print("Zero theta:", info["zero_theta"].numpy())
        print("SOS [b0,b1,b2,a0,a1,a2]:")
        print(info["sos"].numpy())

    slider_f0.on_changed(update)
    slider_q.on_changed(update)
    slider_g.on_changed(update)

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    model = train(CFG)
    interactive_demo(model, CFG)