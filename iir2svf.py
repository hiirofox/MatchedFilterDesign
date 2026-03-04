import sympy as sp

def solve_iir_to_svf(n):
    """
    根据给定的阶数 n，求解 IIR 到 SVF 的系数转换闭式解。
    """
    print(f"正在求解 {n} 阶 IIR 到 SVF 的系数映射...\n")
    
    # 1. 定义符号
    # b0, b1, ..., bn
    b = sp.symbols(f'b0:{n+1}')
    # a0=1, a1, ..., an
    a_syms = sp.symbols(f'a1:{n+1}')
    a = [1] + list(a_syms)
    h = sp.Symbol('h')

    # 2. 将 z = (h-1)/h 代入 H_IIR(z)，并分子分母同乘 h^n 提取多项式
    # 分子 N(h) = sum( b_k * (h-1)^k * h^(n-k) )
    N_h = sum(b[k] * (h - 1)**k * h**(n - k) for k in range(n + 1))
    # 分母 D(h) = sum( a_k * (h-1)^k * h^(n-k) )
    D_h = sum(a[k] * (h - 1)**k * h**(n - k) for k in range(n + 1))

    # 3. 展开多项式
    N_h_exp = sp.expand(N_h)
    D_h_exp = sp.expand(D_h)

    # 4. 提取 h 的各项系数
    N_coeffs = [N_h_exp.coeff(h, k) for k in range(n + 1)]
    D_coeffs = [D_h_exp.coeff(h, k) for k in range(n + 1)]

    # 5. 归一化分母
    # SVF 的分母常数项为 1 (即 1 - c1*h - ...)，所以必须用 D(h) 的常数项 C0 进行归一化
    C0 = D_coeffs[0]
    
    c_sols = {}
    d_sols = {}

    # 6. 求解 c 系数 (注意 SVF 公式里是减号：1 - c1*h - c2*h^2 ...)
    # 所以 1 - c_k * h^k = (D_coeffs[k] / C0) * h^k => c_k = -D_coeffs[k] / C0
    for k in range(1, n + 1):
        c_sols[f'c{k}'] = sp.simplify(-D_coeffs[k] / C0)

    # 7. 求解 d 系数
    # d_k * h^k = (N_coeffs[k] / C0) * h^k => d_k = N_coeffs[k] / C0
    for k in range(n + 1):
        d_sols[f'd{k}'] = sp.simplify(N_coeffs[k] / C0)

    return c_sols, d_sols

# ==========================================
# 测试与输出（以最常用的 2阶 Biquad 为例）
# ==========================================
if __name__ == "__main__":
    
    order = 8  # 你可以改成任意阶数，如 1, 3, 4
    
    c_res, d_res = solve_iir_to_svf(order)

    print("=== 反馈系数 c (对应分母 a 系数) ===")
    for name, expr in c_res.items():
        print(f"{name} = {expr}")

    print("\n=== 前馈系数 d (对应分子 b 系数) ===")
    for name, expr in d_res.items():
        print(f"{name} = {expr}")
        
        
#积分器h=1/(1-z^-1)