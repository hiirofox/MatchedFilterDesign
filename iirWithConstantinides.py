import sympy as sp

def generate_warped_iir_code():
    # 1. 定义符号
    b0, b1, b2, b3, b4 = sp.symbols('b0 b1 b2 b3 b4')
    a0, a1, a2, a3, a4 = sp.symbols('1 a1 a2 a3 a4')
    alpha = sp.symbols('alpha')
    Z = sp.symbols('Z') # 代表 z^-1

    # 2. 定义原始 4 阶传递函数的分子和分母
    num = b0 + b1*Z + b2*Z**2 + b3*Z**3 + b4*Z**4
    den = a0 + a1*Z + a2*Z**2 + a3*Z**3 + a4*Z**4

    # 3. 定义 Constantinides 代换式
    warp_expr = (Z - alpha) / (1 - alpha * Z)

    # 4. 执行代入
    num_sub = num.subs(Z, warp_expr)
    den_sub = den.subs(Z, warp_expr)

    # 5. 消除分母并约分化简 (修复 Z 泄漏的核心)
    clear_factor = (1 - alpha * Z)**4
    num_simplified = sp.cancel(num_sub * clear_factor)
    den_simplified = sp.cancel(den_sub * clear_factor)
    
    # 展开多项式
    num_expanded = sp.expand(num_simplified)
    den_expanded = sp.expand(den_simplified)

    # 6. 按 Z 的幂次收集同类项
    num_coeffs = sp.collect(num_expanded, Z, evaluate=False)
    den_coeffs = sp.collect(den_expanded, Z, evaluate=False)

    # 7. 打印生成的 C/Python 代码
    print("=== 生成的新滤波器系数公式 ===")
    print("// 注意：计算完成后，所有的 B_i 和 A_i 都需要除以 A_0 进行归一化\n")
    
    for i in range(5):
        # 提取对应 Z**i 的系数（Z**0 就是 1）
        power = Z**i if i > 0 else sp.S.One
        
        # 获取展开后的表达式并转换为代码字符串
        B_expr = str(num_coeffs.get(power, 0)).replace('**', '^')
        A_expr = str(den_coeffs.get(power, 0)).replace('**', '^')
        
        # 针对 C/Python 语法的简单格式化 (把 ^ 换回 **)
        B_expr = B_expr.replace('^', '**')
        A_expr = A_expr.replace('^', '**')

        print(f"B_{i} = {B_expr}")
        print(f"A_{i} = {A_expr}\n")

if __name__ == "__main__":
    generate_warped_iir_code()