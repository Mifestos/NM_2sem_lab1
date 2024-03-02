import numpy as np
import pandas as pd
from numpy.linalg import matrix_power
import json


def input_matrices():
    with open('input.json', 'r') as file:
        input_file = json.load(file)
    k = int(input_file['k'])
    e = float(input_file['e'])
    n_matrices = int(input_file['n_matrices'])

    A = [0] * n_matrices
    D = [0] * n_matrices
    C = [0] * n_matrices
    U = [0] * n_matrices
    for i in range(0, n_matrices):
        D[i] = np.array(input_file[f"D{i}"])
        C[i] = np.array(input_file[f"C{i}"])
        U[i] = np.array(input_file[f"U{i}"])
        A[i] = D[i] + k * C[i]
    return A, U, e, n_matrices


def max_eigenvalue(A, U, e, n_matrices, max_iter=100):
    max_values = [0] * n_matrices
    for i in range(0, n_matrices):
        iterations = [0] * max_iter
        j = 0
        while j < max_iter - 1:
            iterations[j] = round(
                np.max(np.dot(matrix_power(A[i], j + 1), U[i])) / np.max(np.dot(matrix_power(A[i], j), U[i])), 4)
            if i > 0 and abs(iterations[j] - iterations[j - 1]) < e:
                break
            j += 1
        max_values[i] = iterations[:j][-1]
    return max_values


def min_eigenvalue(A, U, e, n_matrices, max_iter=100):
    A_inv = [0] * n_matrices
    for i in range(0, n_matrices):
        A_inv[i] = np.linalg.inv(A[i])
    min_values = [0] * n_matrices
    for i in range(0, n_matrices):
        iters = [None] * max_iter
        j = 0
        while j < max_iter - 1:
            iters[j] = round(1 / (np.max(np.dot(matrix_power(A_inv[i], j + 1), U[i])) / np.max(
                np.dot(matrix_power(A_inv[i], j), U[i]))), 4)
            if j > 0 and abs(iters[j] - iters[j - 1]) <= e:
                break
            j += 1

        min_values[i] = iters[:j][-1]
    return min_values


def discrepancy(A, max_values, min_values, n_matrices):
    max_discrepancy = [None] * n_matrices
    min_discrepancy = [None] * n_matrices

    for i in range(n_matrices):
        eigenvalues = np.linalg.eigvals(A[i])
        max_eigenvalue = np.max(eigenvalues)
        min_eigenvalue = np.min(eigenvalues)

        max_discrepancy[i] = max_eigenvalue - max_values[i]
        min_discrepancy[i] = min_eigenvalue - min_values[i]

    return max_discrepancy, min_discrepancy


def export_to_excel(max_values, min_values, max_discrepancy, min_discrepancy, filename):
    df = pd.DataFrame({'Max values': max_values, 'Min values': min_values,
                       'Max discrepancy': max_discrepancy, 'Min discrepancy': min_discrepancy})

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        for column_cells in worksheet.columns:
            length = max(len(str(cell.value)) for cell in column_cells)
            worksheet.column_dimensions[column_cells[0].column_letter].width = length


A, U, e, n_matrices = input_matrices()
max_values = max_eigenvalue(A, U, e, n_matrices, max_iter=100)
min_values = min_eigenvalue(A, U, e, n_matrices, max_iter=100)
max_discrepancy, min_discrepancy = discrepancy(A, max_values, min_values, n_matrices)
export_to_excel(max_values, min_values, max_discrepancy, min_discrepancy, 'result.xlsx')

