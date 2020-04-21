import numpy as np

def _pseudo(doc, n):
    index = np.argpartition(doc, -n)[-n:]
    return index


doc = np.array([9, 4, 4, 3, 3, 9, 0, 4, 6, 0])
rst = _pseudo(doc, 2)
print(rst)
