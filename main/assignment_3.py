
# src/main/assignment_3.py

import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    aug = np.hstack((A.astype(float), b.reshape(-1, 1)))
    for i in range(n):
        for j in range(i + 1, n):
            if aug[i][i] == 0:
                raise ZeroDivisionError("Division by zero.")
            factor = aug[j][i] / aug[i][i]
            aug[j] = aug[j] - factor * aug[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (aug[i][-1] - np.dot(aug[i, i + 1:n], x[i + 1:n])) / aug[i][i]
    return x

def solve_question_1():
    A = np.array([
        [2, -1, 1],
        [1, 3, 1],
        [-1, 5, 4]
    ])
    b = np.array([6, 0, -3])
    x = gaussian_elimination(A, b)
    for xi in x:
        print(round(xi, 4))

def lu_factorization(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j][i] / U[i][i]
            L[j][i] = factor
            U[j] = U[j] - factor * U[i]
    det = np.prod(np.diag(U))
    return L, U, det

def solve_question_2():
    A = np.array([
        [1, 1, 0, 3],
        [2, 1, -1, 1],
        [3, -1, -1, 2],
        [-1, 2, 3, -1]
    ])
    L, U, det = lu_factorization(A)
    print(round(det, 4))
    print(L)
    print(U)

def is_diagonally_dominant(A):
    for i in range(len(A)):
        row_sum = np.sum(np.abs(A[i])) - np.abs(A[i][i])
        if np.abs(A[i][i]) < row_sum:
            return False
    return True

def solve_question_3():
    A = np.array([
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]
    ])
    print(int(is_diagonally_dominant(A)))

def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def solve_question_4():
    A = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ])
    print(int(is_positive_definite(A)))

if __name__ == "__main__":
    solve_question_1()
    solve_question_2()
    solve_question_3()
    solve_question_4()
