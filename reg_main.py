import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
#import math
input_dim=3
mid_dim=2
output_dim=2
matrix_1 = np.random.randn(input_dim,mid_dim)
b_1=np.random.randn(1,mid_dim)
matrix_2 = np.random.randn(mid_dim,output_dim)
b_2=np.random.randn(1,output_dim)

iteration=500
lr=5

def regularL2(data):
    data = normalize(data, norm='l2', axis=0)
    return data

def normali(data):
    data=(data-data.min(axis=0))/ (data.max(axis=0)-data.min(axis=0))
    return data



#file parse
with open("train.txt", "r") as file:
    lines = file.readlines()

# 提取第一行的数据
data = lines[0].strip()
data = data.replace("(", "").replace(")", "")  # 移除括号
data = data.split(",")
data = list(map(lambda x: int(x), data))
data = np.array(data)
train_set_in = data.reshape(80, 3)
#normalization
#normal_input_rec_min=train_set_in.min(axis=0)
#normal_input_rec_max=train_set_in.max(axis=0)
#train_set_in= (train_set_in-train_set_in.min(axis=0))/ (train_set_in.max(axis=0)-train_set_in.min(axis=0))
train_set_in=regularL2(train_set_in)


data1=lines[1].strip()
data1 = data1.replace("(", "").replace(")", "")
data1 = data1.split(",")
data1 = list(map(lambda x: float(x), data1))
data1 = np.array(data1)
train_set_out = data1.reshape(80, 2)
#normalization
#normal_output_rec_min=train_set_out.min(axis=0)
#normal_output_rec_max=train_set_out.max(axis=0)
# train_set_out= (train_set_out-train_set_out.min(axis=0))/ (train_set_out.max(axis=0)-train_set_out.min(axis=0))
train_set_out=regularL2(train_set_out)

with open("test.txt", "r") as file:
    lines = file.readlines()

# 提取第一行的数据
data2 = lines[0].strip()
data2 = data2.replace("(", "").replace(")", "")  # 移除括号
data2 = data2.split(",")
data2 = list(map(lambda x: int(x), data2))
data2 = np.array(data2)
test_set_in = data2.reshape(20, 3)
#normalization
#test_set_in= (test_set_in-normal_input_rec_min)/ (normal_input_rec_max-normal_input_rec_min)
test_set_in=regularL2(test_set_in)

data3=lines[1].strip()
data3 = data3.replace("(", "").replace(")", "")
data3 = data3.split(",")
data3 = list(map(lambda x: float(x), data3))
data3 = np.array(data3)
test_set_out = data3.reshape(20, 2)
#normalization
#test_set_out= (test_set_out-normal_output_rec_min)/ (normal_output_rec_max-normal_output_rec_min)
test_set_out=regularL2(test_set_out)



def sigmoid(x):
    return 1/(1+np.exp(-x))

def forward(input,matrix,b):
    temp1=np.dot(input,matrix)
    temp2=temp1+b
    temp3=sigmoid(temp2)
    return temp3

def loss_cal(y_real,y_out):
    return ((y_real[0][0]-y_out[0][0])**2+(y_real[0][1]-y_out[0][1])**2)/2


train_loss_record=list()
test_loss_record=list()
#forward function test
# input=np.array([[1,2,3]])
# for_1=forward(input,matrix_1,b_1)
# for_2=forward(for_1,matrix_2,b_2)

#train
for iter in range(iteration):
    train_temp_loss=0
    for idx in range(80):
        y_real=train_set_out[idx].reshape(1,2)

        input=train_set_in[idx].reshape(1,3)
        for_1=forward(input,matrix_1,b_1)  #1*2
        for_2=forward(for_1,matrix_2,b_2)  #1*2


        train_temp_loss+=loss_cal(y_real,for_2)
        #renew
        temp_martrix_2 = np.array([matrix_2[0][0], matrix_2[1][1]])
        matrix_2=matrix_2-lr*(for_2-y_real)*(1-for_2)*for_2*for_1.T
        b_2=b_2-lr*(for_2-y_real)*(1-for_2)*for_2

        matrix_1=matrix_1-lr*(for_2-y_real)*(1-for_2)*for_2*temp_martrix_2*(1-for_1)*for_1*input.T
        b_1 = b_1 - lr * (for_2 - y_real) * (1 - for_2) * for_2 * temp_martrix_2 * (1 - for_1) * for_1

        #print(matrix_2[0][0])
    train_temp_loss_mean=train_temp_loss/80
    train_loss_record.append(train_temp_loss_mean)

    #test
    test_temp_loss = 0
    for idx in range(20):
        y_real = test_set_out[idx].reshape(1, 2)

        input = test_set_in[idx].reshape(1, 3)
        for_1 = forward(input, matrix_1, b_1)  # 1*2
        for_2 = forward(for_1, matrix_2, b_2)  # 1*2

        test_temp_loss += loss_cal(y_real, for_2)

    test_temp_loss_mean = test_temp_loss / 20
    test_loss_record.append(test_temp_loss_mean)

plt.plot(test_loss_record, label='test', marker='o', linestyle='-')
plt.plot(train_loss_record, label='train', marker='x', linestyle='--')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('loss curve')

plt.show()



# print(matrix_2)
# print(for_2)
# print(real_out)
# print(loss_cal(real_out,for_2))

#test
test_temp_loss=0
for idx in range(20):
    y_real = test_set_out[idx].reshape(1, 2)

    input = test_set_in[idx].reshape(1, 3)
    for_1 = forward(input, matrix_1, b_1)  # 1*2
    for_2 = forward(for_1, matrix_2, b_2)  # 1*2

    test_temp_loss += loss_cal(y_real, for_2)

test_temp_loss_mean = test_temp_loss / 20
print("test set loss is : "+str(test_temp_loss_mean))