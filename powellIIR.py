import math
import cmath
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

PI = math.pi
TWO_PI = 2.0 * math.pi


def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def sigmoid(x):
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def db20(x):
    return 20.0 * math.log10(max(x, 1e-12))


def lerp(a, b, t):
    return a + (b - a) * t


def make_log_grid(fmin, fmax, n):
    out = []
    la = math.log(fmin)
    lb = math.log(fmax)
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        out.append(math.exp(lerp(la, lb, t)))
    return out


class PowellOptimizer:
    def __init__(self, dim):
        self.dim = dim
        self.x = [0.0] * dim
        self.fx = float("inf")
        self.best_x = [0.0] * dim
        self.best_fx = float("inf")
        self.has_best = False
        self.iter_count = 0
        self.directions = []
        self.reset([0.0] * dim)

    def reset(self, x0):
        self.x = list(x0)
        self.fx = float("inf")
        self.best_x = list(x0)
        self.best_fx = float("inf")
        self.has_best = False
        self.iter_count = 0

        self.directions = []
        for i in range(self.dim):
            d = [0.0] * self.dim
            d[i] = 1.0
            self.directions.append(d)

    def _update_best(self, x, fx):
        if (not self.has_best) or (fx < self.best_fx):
            self.best_x = list(x)
            self.best_fx = fx
            self.has_best = True

    @staticmethod
    def _add_scaled(x, d, alpha):
        out = [0.0] * len(x)
        for i in range(len(x)):
            out[i] = x[i] + alpha * d[i]
        return out

    @staticmethod
    def _sub(a, b):
        out = [0.0] * len(a)
        for i in range(len(a)):
            out[i] = a[i] - b[i]
        return out

    @staticmethod
    def _norm(v):
        s = 0.0
        for x in v:
            s += x * x
        return math.sqrt(s)

    def _line_search_bracket(self, func, x, d, step0=1.0, expand=1.8, max_expand=10):
        f0 = func(x)
        self._update_best(x, f0)

        a = 0.0
        fa = f0

        b = step0
        xb = self._add_scaled(x, d, b)
        fb = func(xb)
        self._update_best(xb, fb)

        if fb >= fa:
            b = -step0
            xb = self._add_scaled(x, d, b)
            fb = func(xb)
            self._update_best(xb, fb)

            if fb >= fa:
                return -step0, 0.0, step0, fb, fa, fb

        c = b * expand
        xc = self._add_scaled(x, d, c)
        fc = func(xc)
        self._update_best(xc, fc)

        k = 0
        while fc < fb and k < max_expand:
            a, fa = b, fb
            b, fb = c, fc
            c = b * expand
            xc = self._add_scaled(x, d, c)
            fc = func(xc)
            self._update_best(xc, fc)
            k += 1

        vals = [(a, fa), (b, fb), (c, fc)]
        vals.sort(key=lambda z: z[0])
        (a, fa), (b, fb), (c, fc) = vals
        return a, b, c, fa, fb, fc

    def _line_search_golden(self, func, x, d, a, c, tol=1e-3, max_iters=16):
        gr = 0.6180339887498949

        x1 = c - gr * (c - a)
        x2 = a + gr * (c - a)

        p1 = self._add_scaled(x, d, x1)
        p2 = self._add_scaled(x, d, x2)
        f1 = func(p1)
        f2 = func(p2)
        self._update_best(p1, f1)
        self._update_best(p2, f2)

        for _ in range(max_iters):
            if abs(c - a) < tol:
                break

            if f1 < f2:
                c = x2
                x2 = x1
                f2 = f1
                x1 = c - gr * (c - a)
                p1 = self._add_scaled(x, d, x1)
                f1 = func(p1)
                self._update_best(p1, f1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (c - a)
                p2 = self._add_scaled(x, d, x2)
                f2 = func(p2)
                self._update_best(p2, f2)

        if f1 < f2:
            return x1, f1
        return x2, f2

    def step(self, func, line_step=1.0, line_tol=1e-3, line_iters=16):
        if self.fx == float("inf"):
            self.fx = func(self.x)
            self._update_best(self.x, self.fx)

        x_start = list(self.x)
        f_start = self.fx

        best_dir_improve = -1.0
        best_dir_idx = 0

        for i in range(self.dim):
            d = self.directions[i]
            a, b, c, fa, fb, fc = self._line_search_bracket(
                func, self.x, d, step0=line_step
            )
            alpha, f_new = self._line_search_golden(
                func, self.x, d, a, c, tol=line_tol, max_iters=line_iters
            )
            x_new = self._add_scaled(self.x, d, alpha)

            improve = self.fx - f_new
            if improve > best_dir_improve:
                best_dir_improve = improve
                best_dir_idx = i

            self.x = x_new
            self.fx = f_new
            self._update_best(self.x, self.fx)

        delta = self._sub(self.x, x_start)
        if self._norm(delta) > 1e-9:
            a, b, c, fa, fb, fc = self._line_search_bracket(
                func, self.x, delta, step0=line_step
            )
            alpha, f_ex = self._line_search_golden(
                func, self.x, delta, a, c, tol=line_tol, max_iters=line_iters
            )
            x_ex = self._add_scaled(self.x, delta, alpha)

            if f_ex < f_start:
                self.directions[best_dir_idx] = delta
                self.x = x_ex
                self.fx = f_ex
                self._update_best(self.x, self.fx)

        self.iter_count += 1
        return self.x, self.fx


class IIR4MagnitudeFitter:
    """
    4阶 IIR:
      2 对共轭极点 + 2 对共轭零点 + 1 个增益

    参数:
      x = [up1, tp1, up2, tp2, uz1, tz1, uz2, tz2, kg]
    映射:
      rp = rp_min + (rp_max-rp_min)*sigmoid(up)
      rz = rz_min + (rz_max-rz_min)*sigmoid(uz)
      theta = pi * sigmoid(t)
      g = exp(kg)
    """

    def __init__(self, fs=48000.0, n_freqs=160):
        self.fs = fs
        self.freqs = make_log_grid(20.0, 24000.0, n_freqs)

        self.rp_min = 0.10
        self.rp_max = 0.9999999999995
        self.rz_min = 0.00
        self.rz_max = 1.60

        self.target_fc = 500.0
        self.target_Q = 1.0
        self.target_gain_db = 6.0
        self.target_stages = 1.0

        self.powell = PowellOptimizer(9)
        self.x0 = [
            2.0, -0.5,
            1.2, 0.8,
            1.0, -0.3,
            0.4, 0.6,
            0.0
        ]
        self.powell.reset(self.x0)

        self.target_mag_db = []
        self.iir_mag_db = []
        self.last_loss = float("inf")

        self.set_target(self.target_fc, self.target_Q, self.target_gain_db, self.target_stages)
        self.iir_mag_db = self.compute_iir_mag_db(self.powell.x)

    def set_target(self, fc, Q, gain_db, stages):
        self.target_fc = clamp(fc, 20.0, 32000.0)
        self.target_Q = clamp(Q, 0.707, 20.0)
        self.target_gain_db = clamp(gain_db, -30.0, 30.0)
        self.target_stages = clamp(stages, 1.0, 3.0)
        self.target_mag_db = self._compute_target_mag_db()

    def reset_optimizer(self, hard=False):
        if hard:
            self.powell.reset(self.x0)
        else:
            seed = self.powell.best_x if self.powell.has_best else self.powell.x
            self.powell.reset(seed)

    def _analog_peaking_mag(self, f_hz):
        fc = self.target_fc
        wc = TWO_PI * fc
        Q = self.target_Q
        stages = self.target_stages
        A_target = 10.0 ** (self.target_gain_db / 20.0)

        w = TWO_PI * f_hz
        B = wc / max(Q, 1e-12)
        x = abs((w * w - wc * wc) / (B * w + 1e-200))
        W = 1.0 / (1.0 + x ** (2.0 * stages))
        mag = 1.0 + (A_target - 1.0) * W
        return mag

    def _compute_target_mag_db(self):
        out = []
        for f in self.freqs:
            out.append(db20(self._analog_peaking_mag(f)))
        return out

    def _decode_params(self, x):
        rp1 = self.rp_min + (self.rp_max - self.rp_min) * sigmoid(x[0])
        tp1 = PI * sigmoid(x[1])

        rp2 = self.rp_min + (self.rp_max - self.rp_min) * sigmoid(x[2])
        tp2 = PI * sigmoid(x[3])

        rz1 = self.rz_min + (self.rz_max - self.rz_min) * sigmoid(x[4])
        tz1 = PI * sigmoid(x[5])

        rz2 = self.rz_min + (self.rz_max - self.rz_min) * sigmoid(x[6])
        tz2 = PI * sigmoid(x[7])

        g = math.exp(x[8])

        pairs = [
            [rp1, tp1],
            [rp2, tp2],
        ]
        pairs.sort(key=lambda it: it[1])
        (rp1, tp1), (rp2, tp2) = pairs

        zpairs = [
            [rz1, tz1],
            [rz2, tz2],
        ]
        zpairs.sort(key=lambda it: it[1])
        (rz1, tz1), (rz2, tz2) = zpairs

        return {
            "rp1": rp1, "tp1": tp1,
            "rp2": rp2, "tp2": tp2,
            "rz1": rz1, "tz1": tz1,
            "rz2": rz2, "tz2": tz2,
            "g": g,
        }

    @staticmethod
    def _pair_factor(radius, theta, z_inv):
        c = math.cos(theta)
        return 1.0 - 2.0 * radius * c * z_inv + (radius * radius) * (z_inv * z_inv)

    def _digital_iir_mag(self, f_hz, x):
        p = self._decode_params(x)
        w = TWO_PI * f_hz / self.fs
        z_inv = cmath.exp(complex(0.0, -w))

        num = p["g"]
        num *= self._pair_factor(p["rz1"], p["tz1"], z_inv)
        num *= self._pair_factor(p["rz2"], p["tz2"], z_inv)

        den = 1.0
        den *= self._pair_factor(p["rp1"], p["tp1"], z_inv)
        den *= self._pair_factor(p["rp2"], p["tp2"], z_inv)

        h = num / den
        return abs(h)

    def compute_iir_mag_db(self, x):
        out = []
        for f in self.freqs:
            out.append(db20(self._digital_iir_mag(f, x)))
        return out

    def loss(self, x):
        p = self._decode_params(x)

        pole_penalty = 0.0
        for rp in (p["rp1"], p["rp2"]):
            d = rp - 0.975
            if d > 0.0:
                pole_penalty += 80.0 * d * d

        gain_reg = 0.002 * (x[8] * x[8])

        s = 0.0
        fc = max(self.target_fc, 1e-6)

        for i, f in enumerate(self.freqs):
            tdb = self.target_mag_db[i]
            ydb = db20(self._digital_iir_mag(f, x))
            e = ydb - tdb

            ratio = max(f, 1e-12) / fc
            lf = math.log(ratio)

            w = 1.0
            w += 0.8 * math.exp(-(lf * lf) / 0.20)   # fc附近
            if f < 120.0:
                w += 0.4
            if f > 12000.0:
                w += 0.3

            s += w * e * e

        return s / len(self.freqs) + pole_penalty + gain_reg

    def optimize_some(self, outer_iters=1):
        for _ in range(outer_iters):
            x_new, fx_new = self.powell.step(
                self.loss,
                line_step=0.8,
                line_tol=1e-3,
                line_iters=12
            )
            self.last_loss = fx_new
        x_show = self.powell.best_x if self.powell.has_best else self.powell.x
        self.iir_mag_db = self.compute_iir_mag_db(x_show)
        return x_show


class IIRFitterApp:
    def __init__(self):
        self.fitter = IIR4MagnitudeFitter(fs=48000.0, n_freqs=150)
        self.dirty = True
        self.running = True

        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_axes([0.08, 0.32, 0.88, 0.63])

        self.ax.set_xscale("log")
        self.ax.set_xlim(20.0, 24000.0)
        self.ax.set_ylim(-40.0, 40.0)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude (dB)")
        self.ax.set_title("4th-order IIR Powell Fitting (single-thread)")
        self.ax.grid(True, which="both", alpha=0.25)

        self.target_line, = self.ax.plot(
            self.fitter.freqs, self.fitter.target_mag_db, label="Analog prototype", linewidth=2
        )
        self.iir_line, = self.ax.plot(
            self.fitter.freqs, self.fitter.iir_mag_db, label="Fitted digital IIR", linewidth=2
        )
        self.ax.legend(loc="upper right")

        self.info_text = self.ax.text(
            0.01, 0.02, "", transform=self.ax.transAxes, va="bottom", ha="left"
        )

        self._build_widgets()
        self._update_plot()

        self.timer = self.fig.canvas.new_timer(interval=20)
        self.timer.add_callback(self._on_timer)
        self.timer.start()

        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _build_widgets(self):
        ax_fc = self.fig.add_axes([0.10, 0.23, 0.78, 0.03])
        ax_q = self.fig.add_axes([0.10, 0.18, 0.78, 0.03])
        ax_gain = self.fig.add_axes([0.10, 0.13, 0.78, 0.03])
        ax_stages = self.fig.add_axes([0.10, 0.08, 0.78, 0.03])

        self.s_fc = Slider(ax_fc, "fc", 20.0, 32000.0, valinit=500.0, valstep=1.0)
        self.s_q = Slider(ax_q, "Q", 0.707, 20.0, valinit=1.0, valstep=0.001)
        self.s_gain = Slider(ax_gain, "gain dB", -30.0, 30.0, valinit=6.0, valstep=0.1)
        self.s_stages = Slider(ax_stages, "stages", 1.0, 3.0, valinit=1.0, valstep=0.01)

        self.s_fc.on_changed(self._on_slider_change)
        self.s_q.on_changed(self._on_slider_change)
        self.s_gain.on_changed(self._on_slider_change)
        self.s_stages.on_changed(self._on_slider_change)

        ax_warm = self.fig.add_axes([0.10, 0.01, 0.15, 0.04])
        ax_hard = self.fig.add_axes([0.28, 0.01, 0.15, 0.04])
        ax_pause = self.fig.add_axes([0.46, 0.01, 0.15, 0.04])

        self.btn_warm = Button(ax_warm, "Warm Reset")
        self.btn_hard = Button(ax_hard, "Hard Reset")
        self.btn_pause = Button(ax_pause, "Pause/Run")

        self.btn_warm.on_clicked(self._on_warm_reset)
        self.btn_hard.on_clicked(self._on_hard_reset)
        self.btn_pause.on_clicked(self._on_pause)

    def _apply_target_from_widgets(self):
        self.fitter.set_target(
            self.s_fc.val,
            self.s_q.val,
            self.s_gain.val,
            self.s_stages.val,
        )
        self.fitter.reset_optimizer(hard=False)
        self.dirty = True

    def _on_slider_change(self, _val):
        self._apply_target_from_widgets()
        self._update_plot()

    def _on_warm_reset(self, _event):
        self.fitter.reset_optimizer(hard=False)
        self.dirty = True
        self._update_plot()

    def _on_hard_reset(self, _event):
        self.fitter.reset_optimizer(hard=True)
        self.dirty = True
        self._update_plot()

    def _on_pause(self, _event):
        self.running = not self.running

    def _on_close(self, _event):
        self.running = False

    def _update_plot(self):
        self.target_line.set_ydata(self.fitter.target_mag_db)
        self.iir_line.set_ydata(self.fitter.iir_mag_db)

        p = self.fitter._decode_params(
            self.fitter.powell.best_x if self.fitter.powell.has_best else self.fitter.powell.x
        )
        info = (
            f"loss={self.fitter.last_loss:.5f}   "
            f"iter={self.fitter.powell.iter_count}   "
            f"stages={self.fitter.target_stages:.2f}   "
            f"rp=({p['rp1']:.4f}, {p['rp2']:.4f})   "
            f"tp=({p['tp1'] * 180.0 / PI:.1f}°, {p['tp2'] * 180.0 / PI:.1f}°)   "
            f"rz=({p['rz1']:.4f}, {p['rz2']:.4f})   "
            f"tz=({p['tz1'] * 180.0 / PI:.1f}°, {p['tz2'] * 180.0 / PI:.1f}°)   "
            f"g={db20(p['g']):.2f} dB   "
            f"{'RUN' if self.running else 'PAUSE'}"
        )
        self.info_text.set_text(info)
        self.fig.canvas.draw_idle()

    def _on_timer(self):
        if self.running and self.dirty:
            self.fitter.optimize_some(outer_iters=1)
            self._update_plot()

    def show(self):
        plt.show()


if __name__ == "__main__":
    app = IIRFitterApp()
    app.show()