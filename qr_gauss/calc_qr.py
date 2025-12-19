import math
import time
import copy
import os

# ---------------------------------------------------------
# 任务 3：QR 分解 
# ---------------------------------------------------------

def write_matrix(filename, name, matrix):
    n = len(matrix)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{name}=...\n")
        f.write("complex([")
        for i, row in enumerate(matrix):
            # QR 任务中主要是实数，但为了兼容 complex 格式:
            val_real = row[i].real if isinstance(row[i], complex) else row[i]
            line = " ".join(f"{val.real:.8f}" if hasattr(val, 'real') else f"{val:.8f}" for val in row)
            f.write(line)
            if i < n - 1: f.write(";\n")
        f.write("],\n[")
        # 虚部全0
        for i, row in enumerate(matrix):
            line = " ".join("0.00000000" for _ in row)
            f.write(line)
            if i < n - 1: f.write(";\n")
        f.write("]);\n")

def mat_sub_norm(A, B):
    n = len(A)
    sum_sq = 0.0
    for i in range(n):
        for j in range(n):
            diff = A[i][j] - B[i][j]
            sum_sq += abs(diff) ** 2
    return math.sqrt(sum_sq)

# --- 读取函数 (内置 Method 解析) ---
def read_matrix_with_method(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None, []

    method = None
    idx_method = content.find("Method")
    if idx_method != -1:
        sub = content[idx_method:]
        eq_idx = sub.find('=')
        if eq_idx != -1:
            end_idx = sub.find(';')
            if end_idx == -1: end_idx = sub.find('\n')
            val_str = sub[eq_idx + 1: end_idx].strip()
            val_str = val_str.replace(')', '').replace('.', '')
            try: method = int(val_str)
            except: pass

    start_bracket = content.find('[')
    end_bracket = content.rfind(']')
    if start_bracket == -1 or end_bracket == -1:
        return method, []

    raw_data = content[start_bracket + 1: end_bracket]
    clean_data = raw_data.replace(';', ' ').replace('\n', ' ')
    all_values = []
    for part in clean_data.split():
        try: all_values.append(float(part))
        except ValueError: pass

    if not all_values: return method, []

    total_len = len(all_values)
    n = int(math.sqrt(total_len))
    if n * n != total_len:
        print(f"Error: Data length {total_len} is not a perfect square.")
        return method, []

    matrix = []
    idx = 0
    for r in range(n):
        row = []
        for c in range(n):
            # 统一转为 float (QR任务是实数)
            row.append(all_values[idx])
            idx += 1
        matrix.append(row)

    return method, matrix

# --- QR 算法 ---

def mat_transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def mat_mul(A, B):
    n = len(A)
    m = len(B[0])
    p = len(B)
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            val = 0.0
            for k in range(p): val += A[i][k] * B[k][j]
            C[i][j] = val
    return C

def mat_vec_mul(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def vec_norm(v):
    return math.sqrt(sum(x * x for x in v))

def qr_method_1_cholesky(A):
    n = len(A)
    AT = mat_transpose(A)
    ATA = mat_mul(AT, A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = ATA[i][i] - s
                if val <= 1e-12: 
                    raise ValueError(f"Method 1 Error: Matrix is singular or not positive definite at index {i} (val={val})")
                if val <= 0: raise ValueError("Matrix not positive definite")
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (ATA[i][j] - s) / L[j][j]
    R = mat_transpose(L)
    R_inv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        R_inv[i][i] = 1.0 / R[i][i]
        for j in range(i + 1, n):
            s = sum(R[i][k] * R_inv[k][j] for k in range(i, j))
            R_inv[i][j] = -s / R[j][j]
    Q = mat_mul(A, R_inv)
    return Q, R

def qr_method_2_householder(A):
    n = len(A)
    R = copy.deepcopy(A)
    Q = [[0.0] * n for _ in range(n)]
    for i in range(n): Q[i][i] = 1.0
    for k in range(n - 1):
        x = [R[i][k] for i in range(k, n)]
        norm_x = vec_norm(x)
        if norm_x < 1e-12: continue
        alpha = -norm_x if x[0] >= 0 else norm_x
        u_vec = x[:]
        u_vec[0] -= alpha
        norm_u = vec_norm(u_vec)
        if norm_u < 1e-12: continue
        u = [val / norm_u for val in u_vec]
        for j in range(k, n):
            dot = sum(u[i - k] * R[i][j] for i in range(k, n))
            for i in range(k, n): R[i][j] -= 2.0 * u[i - k] * dot
        for i in range(n):
            dot = sum(Q[i][r] * u[r - k] for r in range(k, n))
            for r in range(k, n): Q[i][r] -= 2.0 * dot * u[r - k]
    return Q, R

def qr_method_3_givens(A):
    n = len(A)
    R = copy.deepcopy(A)
    Q = [[0.0] * n for _ in range(n)]
    for i in range(n): Q[i][i] = 1.0
    for j in range(n):
        for i in range(n - 1, j, -1):
            if abs(R[i][j]) < 1e-12: continue
            a, b = R[i - 1][j], R[i][j]
            r = math.sqrt(a * a + b * b)
            c, s = a / r, -b / r
            for k in range(j, n):
                t1 = c * R[i - 1][k] - s * R[i][k]
                t2 = s * R[i - 1][k] + c * R[i][k]
                R[i - 1][k], R[i][k] = t1, t2
            for k in range(n):
                t1 = c * Q[k][i - 1] - s * Q[k][i]
                t2 = s * Q[k][i - 1] + c * Q[k][i]
                Q[k][i - 1], Q[k][i] = t1, t2
    return Q, R

def qr_method_4_gram_schmidt(A):
    n = len(A)
    if n == 0: return [], []
    
    # 初始化 Q 和 R
    # Q 用于存放正交基，R 存放系数
    Q = [[0.0] * n for _ in range(n)]
    R = [[0.0] * n for _ in range(n)]

    # 为了方便操作，先将 A 转置为列向量列表
    # p[j] 代表矩阵 A 的第 j 列向量
    p = [[A[r][c] for r in range(n)] for c in range(n)]

    for j in range(n):
        # ---------------------------------------------------------
        # 1. 归一化步骤：计算当前列 p[j] 的范数作为对角元素 R[jj]
        # ---------------------------------------------------------
        norm_p = math.sqrt(sum(x * x for x in p[j]))
        
        # 检查是否发生秩亏 (Rank Deficiency)
        # 如果 norm_p 接近 0，说明当前列 A[:,j] 属于之前基向量的线性组合
        if norm_p < 1e-12:
            R[j][j] = 0.0
            
            # 策略：尝试投影标准基向量 e_k (k=0..n-1) 到当前正交补空间
            found_new_basis = False
            
            for k in range(n):
                # 构造标准基 e_k = [0, ..., 1, ..., 0]
                v = [0.0] * n
                v[k] = 1.0
                
                # 将 e_k 对之前已求出的所有 q_0 ... q_{j-1} 进行正交化
                # v_new = v - sum( <v, q_i> * q_i )
                for i in range(j):
                    # 计算点积 <v, q_i>
                    dot = sum(v[r] * Q[r][i] for r in range(n))
                    # 减去投影
                    for r in range(n):
                        v[r] -= dot * Q[r][i]
                
                # 检查剩下的 v 是否非零
                norm_v = math.sqrt(sum(x * x for x in v))
                if norm_v > 1e-12:
                    # 找到了有效的新方向 归一化并赋值给 Q 的第 j 列
                    for r in range(n):
                        Q[r][j] = v[r] / norm_v
                    found_new_basis = True
                    break
            
            if not found_new_basis:
                # 极罕见情况：无法找到补全向量（通常意味着维度设置错误）
                pass
                
        else:
            # --- 正常情况 ---
            R[j][j] = norm_p
            # q_j = p_j / ||p_j||
            for r in range(n):
                Q[r][j] = p[j][r] / norm_p

        # 将新生成的 q_j 从所有后续的列 p[k] (k > j) 中移除
        for k in range(j + 1, n):
            # 计算投影系数 R[j][k] = <q_j, p_k>
            dot = sum(Q[r][j] * p[k][r] for r in range(n))
            R[j][k] = dot
            
            # 更新后续列：p_k = p_k - <q_j, p_k> * q_j
            for r in range(n):
                p[k][r] -= dot * Q[r][j]

    return Q, R

def check_orthogonality(Q):
    n = len(Q)
    QTQ = mat_mul(mat_transpose(Q), Q)
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    return mat_sub_norm(QTQ, I)

def check_reconstruction(A, Q, R):
    QR = mat_mul(Q, R)
    return mat_sub_norm(A, QR)

def run_qr_logic(method, A, file_suffix, sub_index=None):
    print(f"Running QR Method {method}...")
    t0 = time.time()
    try:
        if method == 1: Q, R = qr_method_1_cholesky(A)
        elif method == 2: Q, R = qr_method_2_householder(A)
        elif method == 3: Q, R = qr_method_3_givens(A)
        elif method == 4: Q, R = qr_method_4_gram_schmidt(A)
        else: raise ValueError("Unknown method")
    except Exception as e:
        print(f"Method {method} failed: {e}")
        return
    duration = time.time() - t0
    err_qr = check_reconstruction(A, Q, R)
    err_orth = check_orthogonality(Q)
    suffix = str(file_suffix)
    if sub_index: suffix += f"_{sub_index}"
    write_matrix(f"./Qmat{suffix}.num", "Q", Q)
    write_matrix(f"./Rmat{suffix}.num", "R", R)
    return duration, err_qr, err_orth

def calc_qr(target=None):
    if target is None:
        target = input("Enter file suffix (e.g. 3 for Amat3.num): ").strip()
    filename = f"./num/Amat{target}.num"
    try:
        method, A = read_matrix_with_method(filename)
        if method is None: return
        n = len(A)
        if len(A[0]) != n:
            print("Error: Matrix is not square.")
            return
        # 复数转实数
        A_real = []
        for r in A:
            row_real = []
            for val in r:
                # 兼容 complex 或 float
                c_val = complex(val) if not isinstance(val, complex) else val
                if abs(c_val.imag) > 1e-8:
                    print("Error: Matrix is not real.")
                    return
                row_real.append(c_val.real)
            A_real.append(row_real)

        results_info = []
        if method > 0:
            res = run_qr_logic(method, A_real, target)
            if res: results_info.append((method, res))
        elif method == 0:
            for m in [1, 2, 3, 4]:
                res = run_qr_logic(m, A_real, target, sub_index=m)
                if res: results_info.append((m, res))
        else:
            print("Error: Invalid Method.")
            return

        with open(f"./Res{target}.num", 'w', encoding='utf-8') as f:
            if method > 0:
                dur, err_qr, err_orth = results_info[0][1]
                f.write(f"Время вычисления QR-разложения: {dur:.6f} секунд.\n")
                f.write(f"Погрешность полученного QR-разложения: {err_qr:.6e}.\n")
                f.write(f"Погрешность ортогональности матрицы Q: {err_orth:.6e}.\n")
            else:
                for m, (dur, err_qr, err_orth) in results_info:
                    f.write(f"Method = {m}\n")
                    f.write(f"Время вычисления QR-разложения: {dur:.6f} секунд.\n")
                    f.write(f"Погрешность полученного QR-разложения: {err_qr:.6e}.\n")
                    f.write(f"Погрешность ортогональности матрицы Q: {err_orth:.6e}.\n")
        print(f"Task 3 completed for {filename}")

    except Exception as e:
        print(f"failed: {e}")

if __name__ == "__main__":
    calc_qr()