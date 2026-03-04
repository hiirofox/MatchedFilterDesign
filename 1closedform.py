import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
def exact_9point_closed_form_generic(b_a, a_a, fs, wc):
    """
    带有温柔引力映射的 32点 超定最小二乘拟合
    alpha: 聚集程度控制。1.0为纯线性，0.0为极度聚集。推荐 0.4 ~ 0.6
    """
    N_points = 2048
    alpha = 1
    # ==========================================
    # 步骤 1：按比例分配与引力映射撒点
    # ==========================================
    w_max = np.pi  # 必须钉死在 Nyquist
    w_d = 2.0 * np.pi * wc / fs
    w_d = np.clip(w_d, 0, np.pi) # 保护极值，防止贴边

    # 1a. 保持线性密度平衡：按频段比例分配 32 个点
    N_left = int(np.round(N_points * (w_d / np.pi)))
    N_left = np.clip(N_left, 6, N_points - 6)  # 保证两边都不会“断粮”
    N_right = N_points - N_left

    # 1b. 定义温柔的映射函数 (Linear blended with Smoothstep)
    def gentle_warp(t):
        return alpha * t + (1.0 - alpha) * (3 * t**2 - 2 * t**3)

    # 1c. 左半段 [0, w_d]
    # endpoint=False 是为了把 w_d 这个交界点留给右半段，防止点重合
    t_left = np.linspace(0, 1.0, N_left, endpoint=False) 
    w_left = w_d * gentle_warp(t_left)

    # 1d. 右半段 [w_d, pi]
    t_right = np.linspace(0, 1.0, N_right)
    w_right = w_d + (w_max - w_d) * gentle_warp(t_right)

    # 拼接！得到极度优雅的 32 个节点
    w_nodes = np.concatenate((w_left, w_right))

    # 计算目标真实能量
    Omega_nodes = w_nodes * fs 
    _, H_a = signal.freqs(b_a, a_a, Omega_nodes)
    R_target = np.abs(H_a)**2

    # ==========================================
    # 步骤 2：加权超定线性方程组 (Weighted Least-Squares)
    # ==========================================
    A = np.zeros((N_points, 9))
    A[:, 0] = 1.0
    for i in range(1, 5):
        A[:, i] = np.cos(i * w_nodes)
        A[:, i+4] = -R_target * np.cos(i * w_nodes)
    
    # --- 魔法加权开始 ---
    # 1. 抵消 Equation Error 的隐性衰减：用 R_target 的倒数作为基础权重
    # 加 1e-6 是为了防止除零，取平方根是因为我们算的是幅度
    W = 1.0 / (np.sqrt(R_target) + 1e-6) 
    
    # 2. 独裁者权重：强行把 Nyquist 那个点（最后一个点）的权重放大 1000 倍！
    # 告诉算法：“哪怕前面全毁了，这个点你也必须给我按在地上！”
    W[-1] *= 1000.0 
    
    # 将权重应用到矩阵 A 和目标向量 R_target 的每一行
    A_weighted = A * W[:, np.newaxis]
    R_weighted = R_target * W
    # --- 魔法加权结束 ---
    
    # 使用加权后的矩阵求解
    coeffs, residuals, rank, s = np.linalg.lstsq(A_weighted, R_weighted, rcond=None)
    
    c = coeffs[0:5]
    d = np.concatenate(([1.0], coeffs[5:9]))
    # ==========================================
    # 步骤 3：多项式降维与代数求根 (费拉里法则替代谱分解)
    # ==========================================
    def algebraic_roots_to_z(poly_cos_coeffs):
        c0, c1, c2, c3, c4 = poly_cos_coeffs
        
        # 切比雪夫展开合并为纯 4 次方程系数
        C4 = 8 * c4
        C3 = 4 * c3
        C2 = 2 * c2 - 8 * c4
        C1 = c1 - 3 * c3
        C0 = c0 - c2 + c4
        
        # 模拟 C 语言中的费拉里代数求根公式
        x_roots = np.roots([C4, C3, C2, C1, C0])
        
        z_roots = []
        for x in x_roots:
            # 闭式反推 z，并选择单位圆内的稳定极点
            z1 = x + np.sqrt(x**2 - 1 + 0j)
            z2 = x - np.sqrt(x**2 - 1 + 0j)
            
            if np.abs(z1) < 1.0:
                z_roots.append(z1)
            else:
                z_roots.append(z2)
                
        return np.real(np.poly(z_roots))

    b_unscaled = algebraic_roots_to_z(c)
    a_final = algebraic_roots_to_z(d)

    # ==========================================
    # 步骤 4：自适应全局增益校准
    # ==========================================
    # 找这 9 个点里能量最大的那个点对齐（兼容峰值、带通、高通等任意器型）
    idx_max = np.argmax(R_target)
    z_ref = np.exp(1j * w_nodes[idx_max])
    
    mag_num = np.abs(np.polyval(b_unscaled, z_ref))
    mag_den = np.abs(np.polyval(a_final, z_ref))
    mag_digital = mag_num / mag_den
    
    mag_analog = np.sqrt(R_target[idx_max])
    b_final = b_unscaled * (mag_analog / (mag_digital + 1e-12))

    return b_final, a_final, w_nodes, R_target

# ================= 测试与绘图对比 =================
if __name__ == "__main__":
    fs = 48000
    
    # 【泛化测试】构建一个 18kHz 的高频 Peaking EQ (峰值均衡器)
    # 提升 12dB，Q=2.0。这是极度考验数字滤波器高频映射能力的器型
    fc = 23000 
    wc = 2 * np.pi * fc
    Q = 10.0
    A_gain = 2.0
    
    # 标准模拟 Peaking EQ 传递函数系数 [s^2, s, 1]
    b_a = [1.0, (wc/Q) * A_gain, wc**2]
    a_a = [1.0, (wc/Q) / A_gain, wc**2]
    
    # 一键算出闭式解
    b_dig, a_dig, w_nodes, R_nodes = exact_9point_closed_form_generic(b_a, a_a, fs, fc)
    
    # 评估频响
    freqs = np.linspace(10, 24000, 1000)
    w_digital = 2 * np.pi * freqs / fs
    _, H_analog = signal.freqs(b_a, a_a, 2 * np.pi * freqs)
    _, h_dig = signal.freqz(b_dig, a_dig, worN=w_digital)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, 20 * np.log10(np.abs(H_analog)), 'k--', linewidth=2, label='Target Analog Peaking EQ (+12dB, 18kHz)')
    plt.plot(freqs, 20 * np.log10(np.abs(h_dig)), 'b-', linewidth=2, label='4th-Order Exact Closed-Form')
    
    freqs_nodes = w_nodes * fs / (2 * np.pi)
    #plt.plot(freqs_nodes, 10 * np.log10(R_nodes), 'ro', markersize=6, label='9 Chebyshev Exact Nodes')
    
    plt.axvline(fc, color='gray', linestyle=':', label=f'Center Freq ({fc} Hz)')
    plt.title('Generic Analog Prototype to 4th-Order Closed-Form IIR', fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude (dB)', fontsize=12)
    plt.xlim(10000, 24000)
    plt.ylim(-5, 15)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    print("=== 通用闭式解计算完毕！===")
    print("模拟原型分子 b_a:", b_a)
    print("模拟原型分母 a_a:", a_a)
    print("-" * 30)
    print("生成的 4 阶数字滤波器系数:")
    print("b =", np.array2string(b_dig, precision=8))
    print("a =", np.array2string(a_dig, precision=8))