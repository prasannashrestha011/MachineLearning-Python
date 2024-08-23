from scipy import sparse
import numpy as np
#Defining the data with matrix containing ones
eye=np.ones(4)
row_indicies=np.arange(4)
col_indicies=np.arange(4)
#
eye_coo=sparse.coo_matrix((eye,(row_indicies,col_indicies)))
print(eye_coo)
#convert the matrix into dense format
eye_dense=eye_coo.toarray()
print(eye_dense)