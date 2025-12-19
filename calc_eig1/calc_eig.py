import math
import time
import copy
import sys

# ---------------------------------------------------------
# 任务 5：QR 分解 
# ---------------------------------------------------------

def read_eig_input(filename):
    """
    读取 Task 5 的输入文件并解析 Method, StopValue, Matrix A。
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        sys.exit(1)

    content_lower = content.lower()

    # --- 解析 Method ---
    method = None
    idx_m = content_lower.find("method=")
    if idx_m != -1:
        sub = content[idx_m:]
        eq_idx = sub.find('=')
        if eq_idx != -1:
            end_idx = -1
            for i, char in enumerate(sub[eq_idx + 1:]):
                if char in [';', '\n', '.', ')']:
                    end_idx = eq_idx + 1 + i
                    break
            if end_idx != -1:
                val_str = sub[eq_idx + 1: end_idx].strip()
                try: method = int(val_str)
                except: pass

    # --- 解析 StopValue ---
    stop_val = None
    idx_s = content_lower.find("stopvalue=")
    if idx_s != -1:
        sub = content[idx_s:]
        eq_idx = sub.find('=')
        if eq_idx != -1:
            end_idx = -1
            for i, char in enumerate(sub[eq_idx + 1:]):
                if char in [';', '\n', ',']:
                    end_idx = eq_idx + 1 + i
                    break
            if end_idx != -1:
                val_str = sub[eq_idx + 1: end_idx].strip()
                try: stop_val = float(val_str)
                except: pass

    # --- 解析矩阵 A ---
    start_bracket = content.find('[')
    end_bracket = content.rfind(']')
    matrix = []
    
    if start_bracket != -1 and end_bracket != -1:
        raw_data = content[start_bracket + 1: end_bracket]
        clean_data = raw_data.replace(';', ' ').replace('\n', ' ')
        all_values = []
        for part in clean_data.split():
            try:
                if part.strip(): all_values.append(float(part))
            except ValueError: pass

        total = len(all_values)
        if total > 0:
            n = int(math.sqrt(total))
            if n * n == total:
                idx = 0
                for r in range(n):
                    row = []
                    for c in range(n):
                        row.append(all_values[idx])
                        idx += 1
                    matrix.append(row)
    
    return method, stop_val, matrix

def validate_input(method, stop_val, A):
    """
    严格校验输入条件
    """
    if method not in [0, 1, 2]:
        print(f"Error: Invalid Method {method}. Must be 0, 1, or 2.")
        sys.exit(1)

    if stop_val is None or stop_val <= 0:
        print(f"Error: Invalid StopValue {stop_val}. Must be > 0.")
        sys.exit(1)

    if not A:
        print("Error: Matrix A is empty or could not be read.")
        sys.exit(1)
        
    n = len(A)
    if len(A[0]) != n:
        print(f"Error: Matrix is not square ({n}x{len(A[0])}).")
        sys.exit(1)

    # 检查自共轭（实对称）
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > 1e-8:
                print("Error: Matrix is not symmetric (self-adjoint).")
                sys.exit(1)

# ---------------------------------------------------------
# 计算迭代次数 N 与 能量判据
# ---------------------------------------------------------

def calculate_max_iterations(n, stop_val):
    """
    公式: (1 - 2/(n(n-1)))^N <= StopValue
    推导: N >= ln(StopValue) / ln(1 - 2/(n(n-1)))
    """
    if n < 2: return 1
    
    # 避免 StopValue >= 1 导致 log 为正或 0 (虽然已校验 >0)
    if stop_val >= 1.0:
        return n * n # 默认给个小值

    term = 1.0 - 2.0 / (n * (n - 1))
    
    # 防止 term <= 0 或 term = 1 (当 n=2 时 term=0，log无定义，需特殊处理)
    if term <= 0:
        # n=2 的情况，实际上只需 1 次旋转即可消除非对角元素
        return 10 
    
    # 计算 N
    # ln(StopValue) 是负数，ln(term) 也是负数
    log_stop = math.log(stop_val)
    log_term = math.log(term)
    
    # 向上取整
    N = int(math.ceil(log_stop / log_term))

    return max(N, n * n)

def jacobi_eigenvalue(A_input, method, stop_val):
    """
    使用 Jacobi 旋转法计算特征值。
    """
    A = [row[:] for row in A_input]
    n = len(A)
    start_time = time.time()

    # --- 1. 计算最大迭代次数 ---
    print(f"DEBUG: Calculated Max Iterations") 
    max_iter = calculate_max_iterations(n, stop_val)
    print(f"DEBUG: Calculated Max Iterations N = {max_iter}") 

    # --- 2. 初始化非对角总能量 S2 ---
    # S2 = sum_{i!=j} a_{ij}^2
    S2 = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                S2 += A[i][j] ** 2

    # --- Method 2 初始化: 辅助向量 r ---
    r_vec = [0.0] * n
    if method == 2:
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i != j: s += A[i][j] ** 2
            r_vec[i] = s

    iterations = 0

    while iterations < max_iter:

        if S2 < stop_val:
            break

        iterations += 1
        print(f"DEBUG: Iteration {iterations}, S2 = {S2:.12f}") 
        
        p, q = 0, 1

        # --- 步骤 1: 选择消元元素 (p, q) ---
        if method == 1:
            # 策略: 全局最大元素
            max_val = -1.0
            for i in range(n):
                for j in range(i + 1, n):
                    # 只需要找绝对值最大的，不需要每次重新算 S2
                    if abs(A[i][j]) > max_val:
                        max_val = abs(A[i][j])
                        p, q = i, j

        elif method == 2:
            # 策略: 最优元素
            max_r = -1.0
            p = 0
            for i in range(n):
                if r_vec[i] > max_r:
                    max_r = r_vec[i]
                    p = i

            max_off = -1.0
            best_col = (p + 1) % n
            for j in range(n):
                if p != j:
                    if abs(A[p][j]) > max_off:
                        max_off = abs(A[p][j])
                        best_col = j
            q = best_col
            if p > q: p, q = q, p

        # 选定的消元元素
        apq = A[p][q]
        
        # 如果当前选出的最大元素极小，甚至小于机器精度，也可以提前跳出
        # 这是一个双重保险
        if abs(apq) < 1e-15:
            # 如果最大元素都这么小，S2 肯定也很小了，但在 Method 2 中不一定
            # 此时可以尝试更新 S2 后 continue 或 break
             pass

        # --- 步骤 2: 计算旋转角度 (c, s) ---
        app = A[p][p]
        aqq = A[q][q]
        diff = app - aqq

        if abs(apq) < 1e-15:
            # 无需旋转
            continue

        if abs(diff) < 1e-15:
            c = s = 1.0 / math.sqrt(2)
        else:
            tau = diff / (2.0 * apq)
            t = 1.0 / (abs(tau) + math.sqrt(1.0 + tau ** 2))
            if tau < 0: t = -t
            c = 1.0 / math.sqrt(1.0 + t ** 2)
            s = t * c

        # --- 步骤 3: 能量 S2 更新 (关键) ---
        # 理论上，每次旋转消去 A[p][q]，总能量减少 2 * A[p][q]^2
        # S2_new = S2_old - 2 * apq^2
        # 注意：由于浮点误差，直接减可能导致 S2 变成负数，需修正
        reduction = 2.0 * (apq ** 2)
        S2 -= reduction
        if S2 < 0: S2 = 0.0

        # --- 步骤 4: 执行旋转更新 A ---
        old_app = app
        
        new_app = c * c * app - 2 * s * c * apq + s * s * aqq
        new_aqq = s * s * app + 2 * s * c * apq + c * c * aqq
        
        A[p][p] = new_app
        A[q][q] = new_aqq
        A[p][q] = 0.0 
        A[q][p] = 0.0

        for k in range(n):
            if k != p and k != q:
                a_pk = A[p][k]
                a_qk = A[q][k]
                new_apk = c * a_pk - s * a_qk
                new_aqk = s * a_pk + c * a_qk
                A[p][k] = A[k][p] = new_apk
                A[q][k] = A[k][q] = new_aqk

        # --- 步骤 5: 更新辅助向量 r (Method 2) ---
        if method == 2:
            # O(1) 更新公式
            term_p = (new_app**2) - (old_app**2) - (apq**2)
            new_r_p = r_vec[p] + term_p
            if new_r_p < 0: new_r_p = 0.0
                
            new_r_q = r_vec[q] + r_vec[p] - new_r_p
            if new_r_q < 0: new_r_q = 0.0
            
            r_vec[p] = new_r_p
            r_vec[q] = new_r_q

    end_time = time.time()
    eigenvalues = [A[i][i] for i in range(n)]
    eigenvalues.sort(key=lambda x: abs(x), reverse=True)

    return end_time - start_time, eigenvalues

# ---------------------------------------------------------
# 3. 主程序逻辑
# ---------------------------------------------------------

def calc_eig():
    target = input("Enter suffix (e.g. 13): ").strip()
    if not target: target = "13"

    filename = f"./num/Amat{target}.num"
    out_filename = f"./Res{target}.num"

    # 读取并校验
    method, stop_val, A = read_eig_input(filename)
    validate_input(method, stop_val, A)

    print(f"Loaded Matrix: Size {len(A)}x{len(A)}, Method={method}, Stop={stop_val}")

    results = []

    if method == 0:
        print("Running Method 1 (Max Element)...")
        t1, eigs1 = jacobi_eigenvalue(A, 1, stop_val)
        results.append((1, t1, eigs1))

        print("Running Method 2 (Optimal Element)...")
        t2, eigs2 = jacobi_eigenvalue(A, 2, stop_val)
        results.append((2, t2, eigs2))
    else:
        print(f"Running Method {method}...")
        t, eigs = jacobi_eigenvalue(A, method, stop_val)
        results.append((method, t, eigs))

    try:
        with open(out_filename, 'w', encoding='utf-8') as f:
            if method == 0:
                for m, dur, vals in results:
                    f.write(f"Method={m}.\n")
                    f.write(f"Время вычисления собственных значений: {dur:.6f} секунд.\n")
                    vals_str = " ".join(f"{v:.6f}" for v in vals)
                    f.write(f"Собственные значения: {vals_str}.\n")
            else:
                m, dur, vals = results[0]
                f.write(f"Method={m}\n")
                f.write(f"Время вычисления собственных значений: {dur:.6f} секунд.\n")
                vals_str = " ".join(f"{v:.6f}" for v in vals)
                f.write(f"Собственные значения: {vals_str}.\n")
        print(f"Success! Results saved to {out_filename}")

    except Exception as e:
        print(f"Error writing output: {e}")

if __name__ == "__main__":
    calc_eig()