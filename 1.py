import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def spectral_factorization(coeffs):
    """
    对余弦多项式 P(w) = c0 + c1*cos(w) + c2*cos(2w) + ... 进行数值谱分解。
    返回提取到单位圆内的多项式系数。
    """
    N = len(coeffs) - 1
    # 1. 构造 2N 阶的对称 Z 域多项式
    poly_z = np.zeros(2 * N + 1)
    poly_z[N] = coeffs[0]  # 中心常数项
    for i in range(1, N + 1):
        poly_z[N - i] = coeffs[i] / 2.0  # z^i
        poly_z[N + i] = coeffs[i] / 2.0  # z^-i

    # 2. 求出所有的 2N 个数值根
    roots = np.roots(poly_z)

    # 3. 反射与提取：按照绝对值大小排序，提取最小的那 N 个根
    sorted_indices = np.argsort(np.abs(roots))
    selected_roots = roots[sorted_indices[:N]]

    # 4. 从圆内的根重建稳定的多项式 (z - r1)(z - r2)...
    factor_poly = np.poly(selected_roots)

    # 丢弃极小的数值计算虚部，保证系数为实数
    return np.real(factor_poly)


def design_matched_iir(w_grid, H_target, order):
    """
    通用有理多项式谱分解 IIR 设计器

    输入:
        w_grid   : 归一化角频率数组，范围 [0, pi] (例如 np.linspace(0, np.pi, num_points))
        H_target : 对应的目标幅度响应绝对值 (例如 np.abs(H_a))
        order    : 期望的数字 IIR 滤波器阶数 N (例如 4)
    输出:
        b, a     : 设计出的数字 IIR 滤波器系数 (Z域)
    """
    N = int(order)
    num_points = len(w_grid)

    # === 第一步：获取目标能量响应 ===
    R_target = np.abs(H_target)**2

    # === 第二步：动态构建最小二乘方程组 A * x = b ===
    # A 矩阵列数 = (N+1)个分子系数 + N个分母系数 = 2N + 1
    A = np.zeros((num_points, 2 * N + 1))
    
    # 分子部分: c0 + c1*cos(w) + ... + cN*cos(Nw)
    A[:, 0] = 1.0  # 对应 c0
    for k in range(1, N + 1):
        A[:, k] = np.cos(k * w_grid)  # 对应 c_k
        
    # 分母部分 (已移项): -R_target * (d1*cos(w) + ... + dN*cos(Nw))
    for k in range(1, N + 1):
        A[:, N + k] = -R_target * np.cos(k * w_grid)  # 对应 d_k

    b_vec = R_target # R_target * 1 (d0=1 的项) 移到等式右侧

    # 求解
    x, residuals, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)

    # 分离 c 和 d 系数
    c = x[0 : N+1]
    d = np.concatenate(([1.0], x[N+1 : 2*N+1]))

    # === 第三步：工程安全兜底（强制非负） ===
    # 动态计算整个频段的 P 和 Q 多项式值
    P_vals = c[0] + sum(c[k] * np.cos(k * w_grid) for k in range(1, N + 1))
    Q_vals = d[0] + sum(d[k] * np.cos(k * w_grid) for k in range(1, N + 1))
    
    if np.min(P_vals) <= 0:
        c[0] += abs(np.min(P_vals)) + 1e-8
    if np.min(Q_vals) <= 0:
        d[0] += abs(np.min(Q_vals)) + 1e-8

    # === 第四步：执行谱分解 ===
    b_unscaled = spectral_factorization(c)
    a_final = spectral_factorization(d)

    # === 第五步：增益校准 ===
    # 选择目标频响能量最大的点进行对齐
    idx_max = np.argmax(R_target)
    w_ref = w_grid[idx_max]
    z_ref = np.exp(1j * w_ref)
    
    mag_num = np.abs(np.polyval(b_unscaled, z_ref))
    mag_den = np.abs(np.polyval(a_final, z_ref))
    mag_digital = mag_num / mag_den
    
    mag_analog = np.sqrt(R_target[idx_max])
    gain_correction = mag_analog / (mag_digital + 1e-12)

    b_final = b_unscaled * gain_correction

    return b_final, a_final

# ================= 测试与绘图 =================
if __name__ == "__main__":
    fs = 48000
    num_points = 4096
    
    # --- 1. 定义我们想要的频段和任意目标频响 ---
    w_grid = np.linspace(0, np.pi*0.999, num_points)
    freqs_hz = w_grid * fs / (2 * np.pi)
    freqs_norm = w_grid / np.pi  # 归一化频率 [0, 1]
    
    # 作为一个例子，我们还是用之前的模拟低通，但这次是在外部生成 H_target
    f0 = 20000  
    w0 = 2 * np.pi * f0
    Q = 20.0
    b_a = [w0**2]
    a_a = [1.0, w0/Q, w0**2]
    _, H_target = signal.freqs(b_a, a_a, freqs_hz * 2 * np.pi)
    
    # 你甚至可以在这里放一个纯手工画的、不规则的 target 数组！
    # 只要保证它和 w_grid 长度一致并且大于 0 即可。
    #H_target = 1 / np.sqrt(1 + (freqs_hz / f0)**4)  # 这是一个更陡峭的低通响应

    # --- 2. 调用通用设计器 ---
    target_order = 4 # 你可以把这里改成 6, 8 试试
    
    b_custom, a_custom = design_matched_iir(w_grid, H_target, target_order)
    
    # --- 3. 评估与绘图 ---
    _, h_custom = signal.freqz(b_custom, a_custom, worN=w_grid)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs_hz, 20 * np.log10(np.abs(H_target)), 'k--', linewidth=2, label='Target Magnitude')
    plt.plot(freqs_hz, 20 * np.log10(np.abs(h_custom)), 'b-', linewidth=2, label=f'Custom Fit ({target_order}th order)')
    
    plt.axvline(f0, color='gray', linestyle=':', label=f'Reference Frequency ({f0} Hz)')
    plt.title(f'Universal Magnitude Fit (N={target_order})', fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude (dB)', fontsize=12)
    plt.xlim(200, 24000)
    plt.ylim(-40, 40)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    print(f"=== 生成的 {target_order} 阶数字滤波器系数 ===")
    print("b =", np.array2string(b_custom, precision=6))
    print("a =", np.array2string(a_custom, precision=6))