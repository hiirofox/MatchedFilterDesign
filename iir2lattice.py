import sympy as sp

def solve_iir_to_lattice(n):
    """
    根据给定的阶数 n，求解 IIR 到 Lattice-Ladder 的系数映射。
    使用标准 DSP z^{-1} 约定（负幂次）。
    """
    print(f"正在求解 {n} 阶 IIR 到 Lattice-Ladder 的系数映射...\n")
    
    # 1. 定义符号
    b = sp.symbols(f'b0:{n+1}')
    a_syms = sp.symbols(f'a1:{n+1}')
    a = [1] + list(a_syms) # a[0] = 1
    
    # K 存储反射系数 (Lattice), v 存储梯形系数 (Ladder)
    K = [0] * (n + 1)
    v = sp.symbols(f'v0:{n+1}')
    
    # A_coeffs[m][k] 代表第 m 阶多项式 A_m(z) 的第 k 个系数
    A_coeffs = [[0] * (n + 1) for _ in range(n + 1)]
    for k in range(n + 1):
        A_coeffs[n][k] = a[k]
        
    # ==========================================
    # 步骤 1: Levinson-Durbin 逆推求解 K 系数
    # ==========================================
    for m in range(n, 0, -1):
        # 反射系数 K_m 等于当前 m 阶多项式的最高次项系数
        K[m] = sp.simplify(A_coeffs[m][m])
        
        # 降阶计算 m-1 阶多项式的系数
        A_coeffs[m-1][0] = 1
        for k in range(1, m):
            # 公式: a_k^(m-1) = (a_k^(m) - K_m * a_{m-k}^(m)) / (1 - K_m^2)
            num = A_coeffs[m][k] - K[m] * A_coeffs[m][m-k]
            den = 1 - K[m]**2
            A_coeffs[m-1][k] = sp.simplify(num / den)

    # ==========================================
    # 步骤 2: 构建后向预测误差多项式 B_m(z)
    # ==========================================
    # B_m(z) = z^{-m} A_m(z) = 反转 A_m(z) 的系数
    B_coeffs = [[0] * (n + 1) for _ in range(n + 1)]
    for m in range(n + 1):
        for k in range(m + 1):
            B_coeffs[m][k] = A_coeffs[m][m-k]
            
    # ==========================================
    # 步骤 3: 求解 Ladder 系数 v_m
    # ==========================================
    # 我们要求 sum(v_m * B_m(z)) == B(z)
    # 建立等式: 对于每个 z^{-k} 的系数，左右两边必须相等
    equations = []
    for k in range(n + 1):
        # 提取等式左边 z^{-k} 的系数
        left_side = sum(v[m] * B_coeffs[m][k] for m in range(k, n + 1))
        # 等式右边就是 b[k]
        right_side = b[k]
        equations.append(sp.Eq(left_side, right_side))
        
    # 解线性方程组求 v
    v_sols = sp.solve(equations, v)
    
    # 将字典形式的结果按 v0, v1... 排序整理并化简
    v_res = {v_sym: sp.simplify(v_sols[v_sym]) for v_sym in v}
    K_res = {f'K{m}': K[m] for m in range(1, n + 1)}
    
    return K_res, v_res

# ==========================================
# 测试与输出（以 2阶 Biquad 为例）
# ==========================================
if __name__ == "__main__":
    order = 2  # 建议先测试 1 或 2，测试更高阶时请耐心等待
    K_res, v_res = solve_iir_to_lattice(order)

    print("=== 反射系数 K (对应分母 a 系数，Lattice部分) ===")
    for name, expr in K_res.items():
        print(f"{name} = {expr}")

    print("\n=== 梯形阶跃系数 v (对应分子 b 系数，Ladder部分) ===")
    for name, expr in v_res.items():
        print(f"{name} = {expr}")