import time
import math
from calc_qr import read_matrix_with_method, mat_vec_mul, mat_transpose, calc_qr

# ---------------------------------------------------------
# 任务 4：QR 解方程 
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
            try: row_vals.append(float(p))
            except ValueError: pass
        if row_vals: rows.append(row_vals)
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

# --- 核心逻辑 ---

def back_substitution(R, y):
    """ 解 Rx = y """
    n = len(R)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(R[i][i]) < 1e-12:
            raise ValueError("Matrix is singular (R[i][i] is 0)")
        s = sum(R[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / R[i][i]
    return x

def qr_gauss(target=None):
    if target is None:
        target = input("Enter file suffix (e.g. 10): ").strip()

    amat_file = f"./num/Amat{target}.num"
    bvec_file = f"./num/bvec{target}.num"
    qmat_file = f"./Qmat{target}.num"
    rmat_file = f"./Rmat{target}.num"

    try:
        method, A = read_matrix_with_method(amat_file)
        b_raw = read_matrix(bvec_file)
        if len(b_raw) == 1:
            b = [val.real for val in b_raw[0]]
        else:
            b = [row[0].real for row in b_raw]

        n = len(A)
        if len(b) != n:
            print("Error: Dimension mismatch.")
            return

        Q_mat, R_mat = None, None
        try:
            Q_mat = read_matrix(qmat_file)
            R_mat = read_matrix(rmat_file)
            Q_mat = [[v.real for v in r] for r in Q_mat]
            R_mat = [[v.real for v in r] for r in R_mat]
            print("Found existing QR decomposition files.")
        except FileNotFoundError:
            print("QR files not found. Running calc_qr...")
            calc_qr(target)
            try:
                Q_mat = read_matrix(qmat_file)
                R_mat = read_matrix(rmat_file)
                Q_mat = [[v.real for v in r] for r in Q_mat]
                R_mat = [[v.real for v in r] for r in R_mat]
            except:
                print("Error: Still cannot find QR files.")
                return

        t0 = time.time()
        QT = mat_transpose(Q_mat)
        y = mat_vec_mul(QT, b)
        x = back_substitution(R_mat, y)
        t1 = time.time()

        x_out = [[complex(val, 0)] for val in x]
        write_matrix(f"./xvec{target}.num", "x", x_out)

        Ax = mat_vec_mul([[v.real for v in r] for r in A], x)
        diff = [Ax[i] - b[i] for i in range(n)]
        err = math.sqrt(sum(d * d for d in diff))

        with open(f"./Res{target}.num", 'w', encoding='utf-8') as f:
            f.write(f"Время решения СЛАУ: {t1 - t0:.6f} секунд.\n")
            f.write(f"Погрешность решения СЛАУ: {err:.6e}.\n")

        print(f"Task 4 completed for {amat_file}")

    except Exception as e:
        print(f"Error in qr_gauss: {e}")

if __name__ == "__main__":
    qr_gauss()