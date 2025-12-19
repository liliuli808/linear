import math
import time
import cmath

# ---------------------------------------------------------
# 任务 2：Cholesky 分解
# ---------------------------------------------------------


def parse_matrix_part(content):
    content = content.strip()
    rows = []
    if content.startswith('[') and content.endswith(']'):
        content = content[1:-1]
    lines = content.split(';')
    for line in lines:
        if not line.strip(): continue
        row_vals = []
        parts = line.split()
        for p in parts:
            try:
                row_vals.append(float(p))
            except ValueError:
                pass
        if row_vals:
            rows.append(row_vals)
    return rows

def read_matrix(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if '=' in text:
        text = text.split('=', 1)[1].strip()
        if text.startswith('...'): text = text[3:].strip()
    
    if text.startswith('complex('):
        inner = text[8:].strip()
        if inner.endswith(');'): inner = inner[:-2]
        split_idx = inner.find('],')
        re_str = inner[:split_idx + 1]
        im_str = inner[split_idx + 2:].strip()
        re_mat = parse_matrix_part(re_str)
        im_mat = parse_matrix_part(im_str)
        n = len(re_mat)
        m = len(re_mat[0])
        mat = []
        for i in range(n):
            row = []
            for j in range(m):
                row.append(complex(re_mat[i][j], im_mat[i][j]))
            mat.append(row)
        return mat
    else:
        if text.endswith(';'): text = text[:-1]
        vals = parse_matrix_part(text)
        return [[complex(x, 0) for x in row] for row in vals]

def write_matrix(filename, name, matrix):
    n = len(matrix)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{name}=...\n")
        f.write("complex([")
        for i, row in enumerate(matrix):
            line = " ".join(f"{val.real:.8f}" for val in row)
            f.write(line)
            if i < n - 1: f.write(";\n")
        f.write("],\n[")
        for i, row in enumerate(matrix):
            line = " ".join(f"{val.imag:.8f}" for val in row)
            f.write(line)
            if i < n - 1: f.write(";\n")
        f.write("]);\n")

def mat_mul_adj_lower(C):
    n = len(C)
    res = [[complex(0, 0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = complex(0, 0)
            limit = min(i, j)
            for k in range(limit + 1):
                s += C[i][k] * C[j][k].conjugate()
            res[i][j] = s
    return res

def mat_sub_norm(A, B):
    n = len(A)
    sum_sq = 0.0
    for i in range(n):
        for j in range(n):
            diff = A[i][j] - B[i][j]
            sum_sq += abs(diff) ** 2
    return math.sqrt(sum_sq)

# --- 核心逻辑 ---

def check_hermitian_and_square(A):
    n = len(A)
    if n == 0 or len(A[0]) != n:
        return False, "Not square"
    for i in range(n):
        for j in range(i, n):
            if abs(A[i][j] - A[j][i].conjugate()) > 1e-8:
                return False, "Not Hermitian"
    return True, ""

def cholesky_decomposition(A):
    n = len(A)
    C = [[complex(0, 0)] * n for _ in range(n)]
    for k in range(n):
        sum_sq = 0.0
        for j in range(k):
            c_val = C[k][j]
            sum_sq += (c_val * c_val.conjugate()).real
        diag_val = A[k][k].real - sum_sq
        if diag_val <= 1e-12:
            raise ValueError("Not positive definite (leading minor <= 0)")
        C[k][k] = complex(math.sqrt(diag_val), 0)
        c_kk_real = C[k][k].real
        for i in range(k + 1, n):
            s = complex(0, 0)
            for j in range(k):
                s += C[i][j] * C[k][j].conjugate()
            C[i][k] = (A[i][k] - s) / c_kk_real
    return C

def calc_chol(file_suffix):
    filename = f"./num/Amat{file_suffix}.num"
    out_c = f"./Cmat{file_suffix}.num"
    out_res = f"./Res{file_suffix}.num"

    try:
        print(f"Reading {filename}...")
        A = read_matrix(filename)
        is_valid, msg = check_hermitian_and_square(A)
        if not is_valid:
            print(f"Error: Matrix A is {msg}")
            return
        
        start_t = time.time()
        try:
            C = cholesky_decomposition(A)
        except ValueError as e:
            print(f"Error during decomposition: {e}")
            return
        end_t = time.time()

        write_matrix(out_c, "C", C)
        A_recon = mat_mul_adj_lower(C)
        err = mat_sub_norm(A, A_recon)

        with open(out_res, "w", encoding="utf-8") as f:
            f.write(f"Время вычисления разложения Холецкого: {end_t - start_t:.6f} секунд.\n")
            f.write(f"Погрешность полученного разложения Холецкого: {err:.6e}.\n")
        print(f"Completed. Outputs: {out_c}, {out_res}")

    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    target = input("Enter file suffix (e.g., '2' for Amat2.num): ").strip()
    if not target: target = "2"
    calc_chol(target)