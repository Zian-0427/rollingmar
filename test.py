num = "12.6466,12.5414,12.7773"

num_list = num.split(",")
num_list = [float(num) for num in num_list]
import numpy as np
num = np.array(num_list)
print(num)
print("%.2f(%.2f)" % (num.mean(), num.std()))