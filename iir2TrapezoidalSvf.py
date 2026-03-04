import sympy as sp

def solve_iir_to_trapezoidal_svf(n):
    """
    根据给定的阶数 n，求解 IIR 到梯形积分 SVF 的系数映射闭式解。
    积分器定义：h = 0.5 * (1+z)/(1-z)
    """
    print(f"正在求解 {n} 阶 IIR 到 梯形积分 SVF 的系数映射...\n")
    
    # 1. 定义符号
    b = sp.symbols(f'b0:{n+1}')
    a_syms = sp.symbols(f'a1:{n+1}')
    a = [1] + list(a_syms) # a[0] = 1
    h = sp.Symbol('h')

    # 2. 将 z = (2h-1)/(2h+1) 代入 H_IIR(z)，并分子分母同乘 (2h+1)^n
    # 分子 N(h) = sum( b_k * (2h-1)^k * (2h+1)^(n-k) )
    N_h = sum(b[k] * (2*h - 1)**k * (2*h + 1)**(n - k) for k in range(n + 1))
    
    # 分母 D(h) = sum( a_k * (2h-1)^k * (2h+1)^(n-k) )
    D_h = sum(a[k] * (2*h - 1)**k * (2*h + 1)**(n - k) for k in range(n + 1))

    # 3. 展开多项式
    N_h_exp = sp.expand(N_h)
    D_h_exp = sp.expand(D_h)

    # 4. 提取 h 的各项系数
    N_coeffs = [N_h_exp.coeff(h, k) for k in range(n + 1)]
    D_coeffs = [D_h_exp.coeff(h, k) for k in range(n + 1)]

    # 5. 归一化分母
    # SVF 的分母常数项为 1 (即 1 - c1*h - ...)，需要用常数项 C0 进行归一化
    C0 = sp.simplify(D_coeffs[0])
    
    c_sols = {}
    d_sols = {}

    # 6. 求解 c 系数 (1 - c_k * h^k = ...)
    for k in range(1, n + 1):
        c_sols[f'c{k}'] = sp.simplify(-D_coeffs[k] / C0)

    # 7. 求解 d 系数 (d_k * h^k = ...)
    for k in range(n + 1):
        d_sols[f'd{k}'] = sp.simplify(N_coeffs[k] / C0)

    return c_sols, d_sols

# ==========================================
# 测试与输出（以 2阶 Biquad 为例）
# ==========================================
if __name__ == "__main__":
    order = 8 
    c_res, d_res = solve_iir_to_trapezoidal_svf(order)

    #print("=== 归一化分母项 C0 ===")
    #print(f"C0 = -a1 + a2 + 1\n") # 注：从多项式展开可以轻易看出 C0 的值

    print("=== 反馈系数 c (对应分母 a 系数) ===")
    for name, expr in c_res.items():
        print(f"{name} = {expr}")

    print("\n=== 前馈系数 d (对应分子 b 系数) ===")
    for name, expr in d_res.items():
        print(f"{name} = {expr}")
        
#积分器h=0.5*(1+z)/(1-z)