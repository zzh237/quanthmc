import numpy as np
re = np.load("sample_result.npy",allow_pickle=True).item(0)
print(re['args'])
