import sympy as sp
import symengine as se
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def design_frequency_pade_symbolic(target_expr, z_inv, M, N):
    """
    使用 SymPy 进行纯频域 Padé 滤波器设计
    核心思想：在 z_inv = 1 (即 w = 0, 对应直流低频) 处展开，完美兼容绝对映射 s = -ln(z^-1)
    """
    order = M + N + 1
    
    # 引入偏移变量 w = z_inv - 1
    w = sp.Symbol('w')
    
    # 将目标表达式从 z_inv 域转换到 w 域
    target_expr_w = target_expr.subs(z_inv, w + 1)
    
    # 在 w = 0 处进行级数展开 (对应原点频率上的频域匹配)
    h_series_w = target_expr_w.series(w, 0, order).removeO()
    
    b_syms = sp.symbols(f'b0:{M+1}') 
    a_syms = sp.symbols(f'a1:{N+1}') 
    
    # 建立原始变量 z_inv 的分子分母多项式
    P = sum(b_syms[k] * z_inv**k for k in range(M+1))
    Q = 1 + sum(a_syms[k] * z_inv**(k+1) for k in range(N))
    
    # 将 P 和 Q 也转换到 w 域
    P_w = P.subs(z_inv, w + 1)
    Q_w = Q.subs(z_inv, w + 1)
    
    # 在 w 域构建误差多项式
    error_expr_w = sp.expand(h_series_w * Q_w - P_w)
    
    # 提取 w 的各阶系数构建线性方程组
    equations = []
    for i in range(order):
        coeff = error_expr_w.coeff(w, i)
        if i == 0:
            coeff = error_expr_w.subs(w, 0)
        equations.append(sp.Eq(coeff, 0))
    
    coefficients = list(b_syms) + list(a_syms)
    
    solution = sp.solve(equations, coefficients)
    
    #raw_solution = se.linsolve(equations, coefficients)
    #solution = dict(zip(coefficients, raw_solution))
    
    return solution, b_syms, a_syms


if __name__ == "__main__":
    z_inv = sp.Symbol('z_inv')
    s = sp.Symbol('s')
    
    # 设定采样时间 T
    T = sp.Symbol('T')
    
    # 定义变量
    w0 = sp.Symbol('w0')
    q = sp.Symbol('q')
    
    # 1. 定义模拟原型滤波器 Hs(s)
    target_Hs = w0*w0 / (s*s + w0*s/q + w0*w0)
    #target_Hs = s*s / (s*s + w0*s/q + w0*w0)
    
    # 2. 使用绝对严格的对数映射！
    # 因为我们在 z_inv=1 处展开，此时 s = -ln(1)/T = 0，完全解析！
    target_H = target_Hs.subs(s, -sp.log(z_inv) / T)
    
    # 设定分子和分母的阶数
    M, N = 4, 4
    
    # 3. 解析求解
    sol, b_syms, a_syms = design_frequency_pade_symbolic(target_H, z_inv, M, N)

    
    # 4. 打印解析表达式
    print(f"=== 分子系数 (FIR 部分, 阶数 M={M}) ===")
    for sym in b_syms:
        expr = sp.simplify(sol.get(sym, 0))
        print(f"{sym} = {expr}")
        
    print(f"\n=== 分母系数 (IIR 部分, 阶数 N={N}, 默认 a0=1) ===")
    for sym in a_syms:
        expr = sp.simplify(sol.get(sym, 0))
        print(f"{sym} = {expr}")
        
        
    #以下是绘制频率响应的代码，使用数值方法
    # ==========================================
    # 以下是绘制频率响应及交互界面的代码
    # ==========================================
    
    Fs = 48000
    T_val = 1.0 / Fs
    
    # 关键步骤：使用 lambdify 将复杂的 SymPy 表达式编译为极速的 Numpy 数值函数
    # 这样在拖动滑动条时，只需要几微秒就能计算出新的系数
    b_funcs = [sp.lambdify((T, w0, q), sol.get(sym, 0), "numpy") for sym in b_syms]
    a_funcs = [sp.lambdify((T, w0, q), sol.get(sym, 0), "numpy") for sym in a_syms]

    # 初始化绘图窗口
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.3) # 底部留出空间放滑动条

    # 生成对数频率轴 (从 10 Hz 到 奈奎斯特频率)
    f = np.logspace(1, np.log10(Fs/2 - 1), 1000)
    omega = 2 * np.pi * f
    z_inv_val = np.exp(-1j * omega * T_val)
    
    # 预编译目标模拟滤波器为 numpy 函数
    target_Hs_func = sp.lambdify((w0, q, s), target_Hs, "numpy")

    # 初始参数
    init_f0 = 1000.0
    init_q = 0.707

    # 绘制初始线条
    line_analog, = ax.semilogx(f, np.zeros_like(f), label='Analog Prototype Hs(s)', linestyle='--', color='blue', linewidth=2)
    line_digital, = ax.semilogx(f, np.zeros_like(f), label=f'Digital Padé Hz(z) (M={M}, N={N})', color='red', alpha=0.8)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Frequency Response Comparison')
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    ax.set_ylim(-40, 40) # 预设 Y 轴范围以防止抖动

    # 配置滑动条 (为了 w0 的对数调节，我们在内部映射 10^val)
    ax_f0 = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_q = plt.axes([0.15, 0.08, 0.7, 0.03])
    
    # 注意 f0 滑动条的范围也是对数的
    slider_f0 = Slider(ax_f0, 'Cutoff f0 (Hz)', np.log10(0.02), np.log10(24000), valinit=np.log10(init_f0))
    slider_q = Slider(ax_q, 'Damping q', 0.1, 5.0, valinit=init_q)

    # 自定义滑动条显示的文本格式
    slider_f0.valtext.set_text(f"{10**slider_f0.val:.1f}")

    def update(val):
        # 1. 获取当前滑动条的数值
        curr_f0 = 10**slider_f0.val
        slider_f0.valtext.set_text(f"{curr_f0:.1f}") # 更新显示的文本
        
        curr_w0 = 2 * np.pi * curr_f0
        curr_q = slider_q.val
        
        # 2. 计算模拟滤波器的理想频响 (dB)
        # 为模拟滤波器创建 numpy 函数用于频率向量计算
        target_Hs_func_val = sp.lambdify((w0, q, s), target_Hs, "numpy")
        mag_Hs = 20 * np.log10(np.abs(target_Hs_func_val(curr_w0, curr_q, 1j * omega)) + 1e-12)
        
        # 3. 将数值代入编译好的系数函数中
        b_vals = [func(T_val, curr_w0, curr_q) for func in b_funcs]
        a_vals = [func(T_val, curr_w0, curr_q) for func in a_funcs]
        
        # 4. 计算数字滤波器的频响 (dB)
        num = np.zeros_like(z_inv_val, dtype=complex)
        for k, b_val in enumerate(b_vals):
            num += b_val * (z_inv_val**k)
            
        den = np.ones_like(z_inv_val, dtype=complex)
        for k, a_val in enumerate(a_vals):
            den += a_val * (z_inv_val**(k+1))
            
        Hz = num / den
        mag_Hz = 20 * np.log10(np.abs(Hz) + 1e-12)
        
        # 5. 更新图表数据
        line_analog.set_ydata(mag_Hs)
        line_digital.set_ydata(mag_Hz)
        
        # 动态调整 Y 轴范围
        #min_y = min(np.min(mag_Hs), np.min(mag_Hz))
        #max_y = max(np.max(mag_Hs), np.max(mag_Hz))
        #ax.set_ylim(max(min_y - 5, -80), max_y + 5)
        
        fig.canvas.draw_idle()

    # 绑定回调函数
    slider_f0.on_changed(update)
    slider_q.on_changed(update)

    # 初始化一次绘图
    update(0)

    plt.show()