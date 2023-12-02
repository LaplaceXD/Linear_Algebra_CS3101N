def convert_matrix_to_dataframe(matrix, columns = []):
    """ Converts a matrix into a dataframe with the given column labels. """
    return { label: list(col) for label, col in zip(columns, zip(*matrix)) }

def convert_dataframe_to_matrix(df):
    """ Converts a dataframe into a row matrix.

    # Example Format:
    sample_dataframe = convert_dataframe_to_matrix({
        "f1": [1, 5, 1, 5, 8],
        "f2": [2, 5, 4, 3, 1],
        "f3": [3, 6, 2, 2, 2],
        "f4": [4, 7, 3, 1, 2]
    })
    
    print(sample_dataframe)
    # Output:
    [
        [1, 2, 3, 4],
        [5, 5, 6, 7],
        [1, 4, 2, 3],
        [5, 3, 2, 1],
        [8, 1, 2, 2]
    ]
    """
    return [list(row) for row in zip(*df.values())]

def parse_arff_to_dataframe(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    
    content = content.split("@data")
    attributes, data = content[0].split("@attribute"), content[1].strip().split("\n")
   
    # Parse labels from the attribute tags, and keep track of numeric
    # data types, which will be used for correcting data types of values later on
    labels = [] 
    is_col_numeric = []
    for col, attr in enumerate(attributes[1:]):
        attr_parts = attr.replace("\n", "").strip().split(" ")
        labels.append(attr_parts[0])
        
        if attr_parts[1] == "numeric":
            is_col_numeric.append(col)
    
    # Parse data rows
    data_rows = []
    for r in data:
        row = []
        for col, d in enumerate(r.split(",")):
            fd = None 
            try:
                # Integers for numeric columns else try to parse them as floats
                fd = int(d) if col in is_col_numeric else float(d) 
            except ValueError:
                # If an error occurs fallback to string value, if it is m then None
                fd = d if d != "m" else None
            row.append(fd)
        
        data_rows.append(row)
    
    # Just the relation tag for the arff
    data_label = attributes[0].replace("@relation", "").replace("\n", "").strip()
    
    # Reorganize parsed data to be stored column wise into a dict, with their key being
    # their label
    return data_label, { l: [d[i] for d in data_rows] for i, l in enumerate(labels) }

def vector_magnitude(vector):
    return sum(v ** 2 for v in vector) ** 0.5

def identity(size):
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]  

def diagonal(matrix):
    return [matrix[i][i] for i in range(len(matrix))]

def transpose(matrix):
    return [list(col) for col in zip(*matrix)]

def subtract(matrix_a, matrix_b):
    assert len(matrix_a) == len(matrix_b) and len(matrix_a[0]) == len(matrix_b[0]), "Matrices not the same shape."
    
    return [[a - b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(matrix_a, matrix_b)]

def scale(matrix, val):
    return [[item * val for item in row] for row in matrix]

def dot(matrix_a, matrix_b):
    assert len(matrix_a[0]) == len(matrix_b) or len(matrix_b[0]) == len(matrix_a), "Dimensions of matrices are incompatible for dot product."
    
    # Swap the two matrices if incorrectly oriented
    # A = m x j, B = j x n
    if len(matrix_a[0]) != len(matrix_b):
        matrix_a, matrix_b = matrix_b, matrix_a
   
    result = []
    m, j, n = len(matrix_a), len(matrix_a[0]), len(matrix_b[0])
    for row in range(m):
        row_result = [sum(matrix_a[row][term] * matrix_b[term][col] for term in range(j)) for col in range(n)]
        result.append(row_result)

    return result

def determinant(matrix):
    n = len(matrix)

    # Base Case: Calculate the 2x2 matrix manually
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for cofactor_col in range(n):
        # Calculate the submatrix by excluding the current row and column
        submatrix = [[matrix[row][col] for col in range(n) if col != cofactor_col] for row in range(1, n)]
        
        # Calculate the determinant recursively
        det += ((-1) ** cofactor_col) * matrix[0][cofactor_col] * determinant(submatrix)

    return det

# Standardize using standard normal distribution
def snd_standardize_list(data):
    assert type(data) == list and len(data) != 0, "List must not be empty"
    
    mean = sum(d for d in data) / len(data)
    sample_std = (sum((mean - d) ** 2 for d in data) / (len(data) - 1)) ** 0.5
    
    return [(d - mean) / sample_std for d in data]

# Calculates the covariance of two lists (population)
def calculate_covariance(list_a, list_b):
    assert type(list_a) == list and type(list_b) == list and len(list_a) == len(list_b), "Lists must be of the same length"

    mean_a = sum(a for a in list_a) / len(list_a)
    mean_b = sum(b for b in list_b) / len(list_b)

    return sum((a - mean_a) * (b - mean_b) for a, b in zip(list_a, list_b)) / len(list_a)

# Calculate eigenvalues and eigenvector values through Jacobi eigenvalue algorithm
# https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
def jacobi_method_eigen(matrix, max_iterations = 5, tolerance = 1.0e-9, diff_tolerance = 1.0e-36):
    def max_off_diag_elem(matrix):
        row, col = 0, 1
        max_elem = matrix[row][col]
        n = len(matrix)

        for r in range(n - 1):
            for c in range(r + 1, n):
                if abs(matrix[r][c]) >= max_elem:
                    max_elem = abs(matrix[r][c])
                    row, col = r, c
        
        return max_elem, row, col

    def mutating_rotation(matrix, a, b, k, l, i, j):
        m_kl = matrix[k][l]
        m_ij = matrix[i][j]

        matrix[k][l] = m_kl - a * (m_ij + b * m_kl)
        matrix[i][j] = m_ij + a * (m_kl - b * m_ij)

    n = len(matrix)
    eigenvectors = identity(n)
    for _ in range(max_iterations * (n ** 2)):
        max_elem, max_elem_row, max_elem_col = max_off_diag_elem(matrix)
    
        if max_elem < tolerance:
            return diagonal(matrix), eigenvectors
        
        diff = matrix[max_elem_col][max_elem_col] - matrix[max_elem_row][max_elem_row]
        
        if max_elem < abs(diff) * diff_tolerance:
            t = max_elem / diff
        else:
            phi = diff / (2.0 * max_elem)
            t = 1.0 / (abs(phi) + (phi ** 2 + 1.0) ** 0.5)
            if phi < 0.0:
                t = -t
        
        c = 1.0 / (t ** 2 + 1.0) ** 0.5
        s = t * c
        tau = s / (1.0 + c)

        matrix[max_elem_row][max_elem_col] = 0.0
        matrix[max_elem_row][max_elem_row] -= t * max_elem
        matrix[max_elem_col][max_elem_col] += t * max_elem
        
        for i in range(max_elem_row): 
            mutating_rotation(matrix, s, tau, i, max_elem_row, i, max_elem_col)
        for i in range(max_elem_row + 1, max_elem_col): 
            mutating_rotation(matrix, s, tau, max_elem_row, i, i, max_elem_col)
        for i in range(max_elem_col + 1, n): 
            mutating_rotation(matrix, s, tau, max_elem_row, i, max_elem_col, i)
        
        for i in range(n):
            mutating_rotation(eigenvectors, s, tau, i, max_elem_row, i, max_elem_col)
    
    raise RuntimeError("Jacobi wasn't able to converge the values")

def pca(data, n_components):
    assert len(data[0]) >= n_components and n_components > 0, f"Components must be between n and {len(data[0])}"

    data_t = transpose(data)
    cov_matrix = [[calculate_covariance(feat_b, feat_a) for feat_b in data_t] for feat_a in data_t]
    
    import numpy as np
    eig = np.linalg.eig(np.array(cov_matrix))
    eig = eig.eigenvectors.tolist()
    top_eigenvectors = transpose(eig)[:n_components]
    
    # eigenvalues, eigenvectors = jacobi_method_eigen([[term for term in row] for row in cov_matrix], max_iterations=1000, tolerance=1.0e-128, diff_tolerance=1.0e-128)
    # sorted_eigen_pairs = sorted(zip(eigenvalues, transpose(eigenvectors)), key=lambda item : item[0], reverse=True) 
    # top_eigenvectors = [vec for _, vec in sorted_eigen_pairs[:n_components]] 
   
    return dot(data, transpose(top_eigenvectors))

def svd()

sample_dataset = {
    "f1": [1, 5, 1, 5, 8],
    "f2": [2, 5, 4, 3, 1],
    "f3": [3, 6, 2, 2, 2],
    "f4": [4, 7, 3, 1, 2]
}

# Standardize each indicator in the list
std_df = { feat_label: snd_standardize_list(data) for feat_label, data in sample_dataset.items() }
data_matrix = convert_dataframe_to_matrix(std_df)

result = pca(data_matrix, 2)
print(result)
