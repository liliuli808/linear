import random
import time
import sys

# ---------------------------------------------------------
# 任务 1：LU 分解 
# ---------------------------------------------------------

def generate_random_complex_matrix(n):
    """生成 n x n 随机复矩阵"""
    mat = []
    for _ in range(n):
        row = []
        for _ in range(n):
            # 随机范围可根据需要调整
            val = complex(random.uniform(-10, 10), random.uniform(-10, 10))
            row.append(val)
        mat.append(row)
    return mat

def lu_decomposition_fast(A_in):
    """
    优化的 LU 分解 (A = LU, L 对角线为 1).
    算法：基于高斯消元法的原地更新 (Row-based Gaussian Elimination).
    """
    n = len(A_in)

    # 检查输入矩阵是否为空或非方阵
    if n == 0:
        raise ValueError("Matrix is empty")
    if len(A_in[0]) != n:
        raise ValueError(f"Matrix is not square ({n}x{len(A_in[0])})")

    # 深拷贝 A，用于原地修改成 U (上三角部分) 和 L 的因子 (下三角部分)
    LU = [row[:] for row in A_in]

    for k in range(n):
        # 1. 检查主元 (Strictly Regular check)
        pivot = LU[k][k]
        if abs(pivot) < 1e-12:
            raise ValueError(f"Pivot too small at {k}, not strictly regular.")

        # 优化技巧：将 pivot 行存为局部变量，减少索引查找
        pivot_row = LU[k]

        # 2. 对第 k 行下方的每一行 i 进行消元
        for i in range(k + 1, n):
            row_i = LU[i]
            # 计算乘数 (L 的元素)
            factor = row_i[k] / pivot
            row_i[k] = factor  # 将 L 的部分直接存在下三角位置

            # 更新行: row_i = row_i - factor * pivot_row
            for j in range(k + 1, n):
                row_i[j] -= factor * pivot_row[j]

    # --- 构造结果矩阵 L 和 U ---
    L = [[complex(0, 0)] * n for _ in range(n)]
    U = [[complex(0, 0)] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = complex(1, 0)  # L 对角线为 1
        for j in range(n):
            if i > j:
                L[i][j] = LU[i][j]  # 下三角部分是 L
            else:
                U[i][j] = LU[i][j]  # 上三角及对角线是 U

    return L, U

def calc_lu():
    print("=== Task 1: LU Decomposition (Standalone) ===")

    try:
        n_str = input("Enter N (natural number > 0): ").strip()
        k_str = input("Enter K (natural number > 0): ").strip()
        N = int(n_str)
        K = int(k_str)
        if N <= 0 or K <= 0:
            raise ValueError("Values must be positive integers.")
    except ValueError:
        print("Error: Invalid input. N and K must be natural numbers (integers > 0).")
        return # 立即退出程序

    print(f"Target: N={N}, K={K}")
    T_array = []

    for n in range(1, N + 1):
        while True:
            valid_k = 0
            total_time = 0.0

            for _ in range(K):
                A = generate_random_complex_matrix(n)
                try:
                    t0 = time.time()
                    lu_decomposition_fast(A)
                    t1 = time.time()
                    total_time += (t1 - t0)
                    valid_k += 1
                except ValueError:
                    pass

            if valid_k > 0:
                avg_time = total_time / valid_k
                T_array.append(avg_time)
                break
            else:
                print(f"n={n}: No regular matrices. Retrying...")

        if n % 50 == 0 or n == N:
            print(f"Processed n={n}, T(n)={T_array[-1]:.6e}s")

    try:
        with open("./results1.num", "w", encoding="utf-8") as f:
            for t in T_array:
                # 每一行写入一个数，不带 T = ...
                f.write(f"{t:.8e}\n")
        print("Done. Saved to ./results1.num")
    except Exception as e:
        print(f"File write error: {e}")

if __name__ == "__main__":
    calc_lu()