

"""
In [1]: import numpy as np

In [2]: arr = np.array([1, 3, 2, 4, 5])

In [3]: arr.argsort()[-3:][::-1]
Out[3]: array([4, 3, 1])
"""

import numpy as np

disti = np.array([0.01, 0.93, 0.80, 0.01, 0.01, 0.02, 0.03])
second_arg = disti.argsort()[-2]

print(second_arg)
