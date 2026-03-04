import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def exact_9point_closed_form_iir(wc, Q, fs):
    """
    4阶 IIR 纯闭式解设计器 (代数插值 + 费拉里法则降维)
    """
    # ==========================================
    # 步骤 1：生成 9 个切比雪夫节点频率 (避免插值振荡)
    # ==========================================
    # 纯代数公式生成最优分布的 9 个点
    k = np.arange(1, 10)
    # x_nodes 在 [-1, 1] 之间分布
    x_nodes = np.cos((2 * k - 1) * np.pi / 18.0) 
    # 映射到 [0, pi] 的角频率
    w_nodes = np.arccos(x_nodes) 
    
    # 计算目标模拟滤波器在这 9 个点上的真实绝对能量 R
    freqs_hz = w_nodes * fs / (2 * np.pi)
    s = 1j * 2 * np.pi * freqs_hz
    Hs = wc**2 / (s**2 + s * wc / Q + wc**2)
    R_target = np.abs(Hs)**2

    # ==========================================
    # 步骤 2：解 9x9 线性方程组 (完全闭式的克莱姆法则等效)
    # P(w) - R(w)Q(w) = 0
    # ==========================================
    A = np.zeros((9, 9))
    A[:, 0] = 1.0
    for i in range(1, 5):
        A[:, i] = np.cos(i * w_nodes)
        A[:, i+4] = -R_target * np.cos(i * w_nodes)
    
    # 求解 A * [c, d]^T = R_target
    # 在 C 语言中，这是一个硬编码的 9x9 高斯消元或伴随矩阵公式，无迭代
    coeffs = np.linalg.solve(A, R_target)
    
    c = coeffs[0:5]
    d = np.concatenate(([1.0], coeffs[5:9]))

    # === 安全性兜底（可选，理论上精确插值不需要，但为了防浮点误差）===
    # 注意：闭式解的代价是，如果这 9 个点算出来的响应在其他地方穿零，求根就会得到复数
    
    # ==========================================
    # 步骤 3：多项式降维与代数求根 (替代谱分解)
    # ==========================================
    def algebraic_roots_to_z(poly_cos_coeffs):
        c0, c1, c2, c3, c4 = poly_cos_coeffs
        
        # 使用切比雪夫公式展开，合并同类项，得到纯 4 次方程 C4*x^4 + ... = 0
        C4 = 8 * c4
        C3 = 4 * c3
        C2 = 2 * c2 - 8 * c4
        C1 = c1 - 3 * c3
        C0 = c0 - c2 + c4
        
        # 【核心闭式解】：这里用 np.roots 模拟费拉里四次求根公式
        # 在 C/C++ 中，你可以直接把费拉里公式写在这里，完全不需要调用库
        x_roots = np.roots([C4, C3, C2, C1, C0])
        
        z_roots = []
        for x in x_roots:
            # 闭式解反推 z: z^2 - 2xz + 1 = 0
            # 使用求根公式，并强制选择绝对值（模）小于 1 的那个根，保证系统稳定！
            z1 = x + np.sqrt(x**2 - 1 + 0j)
            z2 = x - np.sqrt(x**2 - 1 + 0j)
            
            if np.abs(z1) < 1.0:
                z_roots.append(z1)
            else:
                z_roots.append(z2)
                
        # 展开根得到多项式系数：(z-z1)(z-z2)(z-z3)(z-z4)
        return np.real(np.poly(z_roots))

    b_unscaled = algebraic_roots_to_z(c)
    a_final = algebraic_roots_to_z(d)

    # ==========================================
    # 步骤 4：全局增益校准
    # ==========================================
    # 用第一个频率点 (w_nodes[0]) 进行增益对齐
    z_ref = np.exp(1j * w_nodes[0])
    mag_num = np.abs(np.polyval(b_unscaled, z_ref))
    mag_den = np.abs(np.polyval(a_final, z_ref))
    mag_digital = mag_num / mag_den
    
    mag_analog = np.sqrt(R_target[0])
    b_final = b_unscaled * (mag_analog / mag_digital)

    return b_final, a_final, w_nodes, R_target

# ================= 测试与绘图对比 =================
if __name__ == "__main__":
    fs = 48000
    fc = 22000 
    wc = 2 * np.pi * fc
    Q = 20  # 适度谐振
    
    # 计算闭式解
    b_dig, a_dig, w_nodes, R_nodes = exact_9point_closed_form_iir(wc, Q, fs)
    
    # 评估高精度频响
    freqs = np.linspace(10, 24000, 1000)
    w_digital = 2 * np.pi * freqs / fs
    s = 1j * 2 * np.pi * freqs 
    Hs = wc**2 / (s**2 + s * wc / Q + wc**2)
    mag_target = np.abs(Hs)
    
    _, h_dig = signal.freqz(b_dig, a_dig, worN=w_digital)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, 20 * np.log10(mag_target), 'k--', linewidth=2, label='Target Analog Prototype')
    plt.plot(freqs, 20 * np.log10(np.abs(h_dig)), 'b-', linewidth=2, label='4th-Order Exact Closed-Form')
    
    # 标出我们用来硬解方程的 9 个采样点
    freqs_nodes = w_nodes * fs / (2 * np.pi)
    plt.plot(freqs_nodes, 10 * np.log10(R_nodes), 'ro', markersize=6, label='9 Exact Interpolation Points')
    
    plt.axvline(fc, color='gray', linestyle=':', label=f'Cutoff ({fc} Hz)')
    plt.title('9-Point Exact Interpolation Closed-Form Filter', fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude (dB)', fontsize=12)
    plt.xlim(100, 24000)
    plt.ylim(-30, 30)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    print("=== 纯代数闭式解求出的 4 阶数字滤波器系数 ===")
    print("b =", np.array2string(b_dig, precision=6))
    print("a =", np.array2string(a_dig, precision=6))