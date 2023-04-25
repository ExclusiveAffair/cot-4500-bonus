import numpy as np

def print_double_spaced(s):
    print(s)
    print()

def print_float_double_spaced(s):
    print("%.5f" % s)
    print()

# returns the divided differences for Hermite polynomial interpolation given a series of (x, y) pairs
def gen_hermite_divided_differences(xs, ys, derivs):
    n = len(xs)
    differences_matrix = np.zeros((2 * n, 2 * n))

    for i in range(2 * n):
        differences_matrix[i][0] = xs[i//2]
        differences_matrix[i][1] = ys[i//2]

    for i in range(1, 2 * n, 2):
        differences_matrix[i][2] = derivs[(i-1)//2]
    
    for i in range(1, 2 * n):
        for j in range(2, i+2):
            if j == 2 and i % 2 == 1: continue
            if j >= 2 * n: continue
            differences_matrix[i][j] = (differences_matrix[i][j-1] - differences_matrix[i-1][j-1]) / (xs[i//2] - xs[(i-j+1)//2])

    return differences_matrix

def make_diagonally_dominant(matrix, b_vector):
    n = len(matrix)

    for i in range(n):
        pivot: float = matrix[i][i]
        sum_of_other_elements = sum(abs(matrix[i][i+1:]))

        # we can guarantee this pivot is the largest in the row
        if abs(pivot) > abs(sum_of_other_elements):
            continue

        # if we reach this point, this means we need to swap AT LEAST ONCE
        max_value_of_row = 0
        max_index_in_row = 0
        for j in range(n):
            current_value_in_row: float = abs(matrix[i][j])
            if current_value_in_row > max_value_of_row:
                max_value_of_row = current_value_in_row
                max_index_in_row = j

        # now that we have a new "pivot", we swap cur_row with the expected index
        matrix[[i, max_index_in_row]] = matrix[[max_index_in_row, i]]
        b_vector[[i, max_index_in_row]] = b_vector[[max_index_in_row, i]]
        
    return matrix, b_vector

def solve_matrix_equation(a, b, x0, tolerance, iter, use_latest):
    n = len(x0)
    cur = x0

    for ct in range(iter):
        cur2 = np.zeros(n)
        for i in range(n):
            cur2[i] = b[i]
            for j in range(0, n):
                if i == j:
                    continue
                xval = (cur2[j] if (use_latest and j < i) else cur[j])
                cur2[i] -= a[i][j] * xval

            cur2[i] /= a[i][i]

        if (np.amax(np.absolute(np.subtract(cur2, cur))) < tolerance):
            return ct + 1
        cur = cur2

    return iter

def jacobi(a, b, x0, tolerance, iter):
    return solve_matrix_equation(a, b, x0, tolerance, iter, False)

def gauss_seidel(a, b, x0, tolerance, iter):
    return solve_matrix_equation(a, b, x0, tolerance, iter, True)

def calc_fx(x, f):
    return eval(f)

def newton_raphson_method(function, derivative, tolerance, p0, max_iterations):
    p_prev = p0
    iter = 0
    while iter < max_iterations:
        iter += 1
        f_prime = calc_fx(p_prev, derivative)

        if (f_prime != 0):
            p_next = p_prev - calc_fx(p_prev, function) / calc_fx(p_prev, derivative)
            if abs(p_next - p_prev) < tolerance:
                return iter
            p_prev = p_next
        else:
            return -1

    return iter

def func(t, y):
    return y - t * t * t

def modified_euler(a, b, y0, n, f):
    h = (b - a) / n
    cur_x = a
    cur_y = y0

    for _ in range(n):
        cur_y = cur_y + (h / 2) * (f(cur_x, cur_y) + f(cur_x + h, cur_y + h * f(cur_x, cur_y)))
        cur_x = cur_x + h
    
    return cur_y

if __name__ == "__main__":
    A = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]])
    b = np.array([1, 3, 0])
    x0 = np.array([0, 0, 0])
    tolerance = 9e-7
    iter = 50
    nA, nb = make_diagonally_dominant(A, b)

    print_double_spaced(gauss_seidel(nA, nb, x0, tolerance, iter))
    print_double_spaced(jacobi(nA, nb, x0, tolerance, iter))

    print_double_spaced(newton_raphson_method("x ** 3 - (x ** 2) + 2", "3 * (x ** 2) - 2 * x", 1e-6, 0.5, 100))

    print_double_spaced(gen_hermite_divided_differences([0, 1, 2], [1, 2, 4], [1.06, 1.23, 1.55]))

    print_float_double_spaced(modified_euler(0, 3, 0.5, 100, func))