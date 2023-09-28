# import numpy as np
#
# # 打开文件以读取数据
# with open("test.txt", "r") as file:
#     lines = file.readlines()
#
# # 提取第一行的数据
# data = lines[0].strip()
# data = data.replace("(", "").replace(")", "")  # 移除括号
# data = data.split(",")
# data = np.array(data)
# train_set_in = data.reshape(20, 3)
#
#
# data1=lines[1].strip()
# data1 = data1.replace("(", "").replace(")", "")
# data1 = data1.split(",")
# data1 = np.array(data1)
# train_set_out = data1.reshape(20, 2)

import numpy as np
from sklearn.preprocessing import normalize

# normalized_data = normalize(data, norm='l2', axis=0)
# print(normalized_data)

def norm(data):
    data = normalize(data, norm='l2', axis=0)
    return data

data = np.array([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0]])

data=norm(data)
print(data)