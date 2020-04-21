import numpy as np

def _rocchio(query, relevant, irrelevant, a, b, v):
    if (query.shape[0] != relevant.shape[1] or relevant.shape[1] != irrelevant.shape[1]):
        print("run failed: incorrest shape")
    mean_rel = np.mean(relevant, axis=0)
    mean_irrel = np.mean(irrelevant, axis=0)
    print(mean_rel)
    print(mean_irrel)
    return a*query + b*mean_rel - v*mean_irrel

query = np.array([1,0,1])
relevant = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
irrelevant = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

rst = _rocchio(query, relevant, irrelevant, 1,1,1)
print(rst)
