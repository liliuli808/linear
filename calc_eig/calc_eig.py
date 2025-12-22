import math
import sys
import time

# --- 1. 文件读取与辅助功能 ---

def read_file(filename):
    """
    读取 .num 文件。
    解析 Method, StopValue 和矩阵 A。
    支持格式：Method=..., StopValue=..., A=[...]
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        sys.exit(1)

    # 简单的解析逻辑
    # 移除换行以便处理
    content = content.replace('\n', ' ')
    
    # 解析 Method
    method = None
    if "Method=" in content:
        try:
            # 提取 Method= 后的数字
            import re
            m_match = re.search(r'Method\s*=\s*(\d+)', content)
            if m_match:
                method = int(m_match.group(1))
        except:
            pass
            
    # 解析 StopValue
    stop_value = None
    # 尝试查找 StopValue= 或者是文件头部的一个浮点数
    import re
    sv_match = re.search(r'StopValue\s*=\s*([0-9\.]+e?[+\-]?[0-9]*)', content)
    if sv_match:
        stop_value = float(sv_match.group(1))
    else:
        # 如果没有显式的 StopValue=，尝试寻找非 Method 的浮点数
        tokens = content.replace(';', ' ').replace('[', ' ').replace(']', ' ').split()
        for t in tokens:
            if "Method" in t or "A" in t or "=" in t: continue
            try:
                val = float(t)
                # 假设很小的数或者特定的浮点数是 StopValue
                if val != method:
                    stop_value = val
                    break
            except:
                continue
    
    if stop_value is None:
        stop_value = 1e-9 # 默认值，防止报错
        
    # 解析矩阵 A
    # 寻找方括号 [...] 之间的内容
    start = content.find('[')
    end = content.rfind(']')
    
    if start == -1 or end == -1:
        print("Error: Matrix format incorrect.")
        sys.exit(1)
        
    matrix_str = content[start+1:end]
    
    # 处理 "complex" 关键字的情况（虽然任务5是实矩阵，但通用读取器应处理）
    # 简单的处理：如果包含 complex，我们假设它是实对称的，只取实部（即第一个括号内容）
    if "complex" in content[:start]:
        # 重新定位到第一个括号
        matrix_str = content[content.find('(')+1 : content.find('),')]
        # 去掉可能的内部方括号
        matrix_str = matrix_str.replace('[', '').replace(']', '')

    rows = matrix_str.split(';')
    matrix = []
    for row_str in rows:
        if not row_str.strip(): continue
        # 解析数字
        vals = []
        for x in row_str.split():
            if x.strip():
                vals.append(float(x))
        if vals:
            matrix.append(vals)
            
    return method, stop_value, matrix

def check_matrix(matrix):
    """检查矩阵是否为方阵且对称 [cite: 244]"""
    n = len(matrix)
    if n == 0: return False
    for row in matrix:
        if len(row) != n: return False
    
    # 检查对称性
    for i in range(n):
        for j in range(i+1, n):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-8:
                return False
    return True

# --- 2. 核心算法实现 ---

def calculate_row_squares(matrix, n):
    """计算每一行非对角元素的平方和 (用于 Method 2)"""
    r = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i != j:
                s += matrix[i][j]**2
        r[i] = s
    return r

def jacobi_method(matrix_input, method_variant, stop_value):
    """
    迭代旋转法求解特征值。
    method_variant: 1 (最大元素) 或 2 (最优元素)
    """
    n = len(matrix_input)
    # 深度拷贝矩阵，避免修改原数据
    A = [row[:] for row in matrix_input]
    
    # 如果是方法2，预先计算 r [cite: 249]
    r_vals = []
    if method_variant == 2:
        r_vals = calculate_row_squares(A, n)

    iteration = 0
    max_iter = n * n * 500 # 防止死循环的安全上限

    while iteration < max_iter:
        k = -1
        l = -1
        max_val = 0.0
        
        # --- 步骤 1: 选择消去元素 a_kl ---
        
        if method_variant == 1:
            # 方法 1: 寻找全矩阵中模最大的非对角元素 [cite: 248]
            for i in range(n):
                for j in range(i + 1, n):
                    val = abs(A[i][j])
                    if val > max_val:
                        max_val = val
                        k, l = i, j
                        
        elif method_variant == 2:
            # 方法 2: 选择最优元素 [cite: 249]
            # 1. 找到 r_i 最大的行
            best_row = -1
            max_r = -1.0
            for i in range(n):
                if r_vals[i] > max_r:
                    max_r = r_vals[i]
                    best_row = i
            
            # 2. 在该行中找最大的元素
            if best_row != -1:
                row_max = -1.0
                best_col = -1
                for j in range(n):
                    if j != best_row:
                        val = abs(A[best_row][j])
                        if val > row_max:
                            row_max = val
                            best_col = j
                
                # 确保 k < l
                if best_row < best_col:
                    k, l = best_row, best_col
                else:
                    k, l = best_col, best_row
                max_val = row_max

        # --- 步骤 2: 检查停止条件 ---
        # 如果当前最大的非对角元素小于阈值，则停止 [cite: 251]
        if max_val < stop_value:
            break
            
        # --- 步骤 3: 计算旋转参数 ---
        # 使用 Lecture 12 第 10 页的公式 
        if abs(A[k][k] - A[l][l]) < 1e-15:
            # 对角元素相等，旋转角为 45 度
            alpha = 1.0 / math.sqrt(2) # cos
            beta = 1.0 / math.sqrt(2)  # sin
        else:
            mu = (2 * A[k][l]) / (A[k][k] - A[l][l])
            # alpha = cos(phi)
            alpha = math.sqrt(0.5 * (1 + 1 / math.sqrt(1 + mu**2)))
            # beta = sin(phi), 注意符号由 mu 决定
            sign_mu = 1.0 if mu >= 0 else -1.0
            beta = sign_mu * math.sqrt(0.5 * (1 - 1 / math.sqrt(1 + mu**2)))

        # --- 步骤 4: 执行旋转更新矩阵 ---
        # 暂存需要用到的旧值
        akk = A[k][k]
        all_val = A[l][l]
        akl = A[k][l]

        # 更新 k 行和 l 行 (以及对应的列，因为是对称的)
        for i in range(n):
            if i != k and i != l:
                aki = A[k][i]
                ali = A[l][i]
                
                # 旋转公式
                # b_ik = c * a_ik + s * a_il
                # b_il = -s * a_ik + c * a_il
                new_ki = aki * alpha + ali * beta
                new_li = -aki * beta + ali * alpha
                
                A[k][i] = new_ki
                A[i][k] = new_ki # 对称更新
                A[l][i] = new_li
                A[i][l] = new_li # 对称更新
        
        # 更新对角线元素和交叉点 (a_kl 变为 0)
        A[k][k] = akk * (alpha**2) + all_val * (beta**2) + 2 * akl * alpha * beta
        A[l][l] = akk * (beta**2) + all_val * (alpha**2) - 2 * akl * alpha * beta
        A[k][l] = 0.0
        A[l][k] = 0.0

        # --- 步骤 5: 更新 r 数组 (仅针对 Method 2) ---
        if method_variant == 2:
            # 只有第 k 行和第 l 行的 r 值发生了显著变化
            # 根据讲义公式 [cite: 130] 或直接重新求和这两行
            sum_k = 0.0
            sum_l = 0.0
            for j in range(n):
                if j != k: sum_k += A[k][j]**2
                if j != l: sum_l += A[l][j]**2
            r_vals[k] = sum_k
            r_vals[l] = sum_l

        iteration += 1

    # 返回对角线元素作为特征值
    eigenvalues = [A[i][i] for i in range(n)]
    return eigenvalues

def bubble_sort_desc(arr):
    """按模从大到小排序 (不允许使用内置 sort) [cite: 247]"""
    res = arr[:]
    m = len(res)
    for i in range(m):
        for j in range(0, m - i - 1):
            if abs(res[j]) < abs(res[j + 1]):
                res[j], res[j + 1] = res[j + 1], res[j]
    return res

# --- 3. 主程序逻辑 ---

def main():
    # 默认处理 Amat13.num，也可通过命令行参数传入
    input_filename = "Amat13.num"
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]

    # 生成输出文件名 Res*.num
    import re
    num_match = re.search(r'Amat(\d+)\.num', input_filename)
    if num_match:
        out_filename = f"Res{num_match.group(1)}.num"
    else:
        out_filename = "Res_output.num"

    # 读取与检查
    method, stop_value, matrix = read_file(input_filename)
    
    if not check_matrix(matrix):
        print("Error: Matrix must be square and symmetric (self-adjoint).")
        sys.exit(1)
    
    if method not in [0, 1, 2]:
        print("Error: Method must be 0, 1, or 2.")
        sys.exit(1)

    if stop_value <= 0:
        print("Error: StopValue must be > 0.")
        sys.exit(1)

    # 准备运行的方法列表
    methods_to_run = []
    if method == 0:
        methods_to_run = [1, 2] # 比较两种方法 [cite: 250]
    else:
        methods_to_run = [method]

    output_lines = []

    for m in methods_to_run:
        # 如果是 Method 0，需要打印子标题
        if method == 0:
            output_lines.append(f"Method={m}.")
        else:
            output_lines.append(f"Method={m}")

        # 计时开始
        start_time = time.time()
        
        # 计算特征值
        evals = jacobi_method(matrix, m, stop_value)
        
        # 计时结束
        end_time = time.time()
        duration = end_time - start_time

        # 排序
        sorted_evals = bubble_sort_desc(evals)

        # 格式化输出字符串
        evals_str = " ".join([f"{x:.6f}" for x in sorted_evals]) # 保留小数位更整洁
        
        output_lines.append(f"Время вычисления собственных значений: {duration:.6f} секунд.")
        output_lines.append(f"Собственные значения: {evals_str}.")

    # 写入文件
    with open(out_filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
        
    print(f"完成。结果已保存至 {out_filename}")

if __name__ == "__main__":
    main()