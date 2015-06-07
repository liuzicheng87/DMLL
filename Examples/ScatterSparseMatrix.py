import DMLL

root = 0

if DMLL.rank == root:
    a = DMLL.scipy.sparse.csr_matrix(DMLL.np.arange(100).astype(DMLL.np.double).reshape(10,10))
    b = DMLL.ScatterSparse(a, root)
    del a
else:
    b = DMLL.ScatterSparse(root)

print b.data