from scipy import sparse
import numpy as np
eye=np.eye(5)
print("original matrix: ",eye)
sparse_matrix=sparse.csr_matrix(eye)
print(sparse_matrix)