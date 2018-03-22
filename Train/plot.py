import matplotlib.pyplot as plt
import numpy as np

infile = r""


train_acc = []
train_acc_key = ["Train-accuracy"]
train_acc_key = ["Train-accuracy"]

val_acc = []
val_acc_key = ["Validation-accuracy"]
val_acc_key = ["Validation-accuracy"]

train_ce = []
train_ce_key = ["Train-cross-entropy"]
val_ce = []
val_ce_key = ["Validation-cross-entropy"]


with open(infile) as f:
    f = f.readlines()

for line in f:
    for phrase in train_acc_key:
        if phrase in line:
            idx = line.index('=')
            train_acc.append(line[idx+1:len(line)-1])
    for phrase in val_acc_key:
        if phrase in line:
            idx = line.index('=')
            val_acc.append(line[idx+1:len(line)-1])
    for phrase in train_ce_key:
        if phrase in line:
            idx = line.index('=')
            train_ce.append(line[idx+1:len(line)-1])
    for phrase in val_ce_key:
        if phrase in line:
            idx = line.index('=')
            val_ce.append(line[idx+1:len(line)-1])
print(max(val_acc)) 


myarray = np.asarray(val_acc)
train_array = np.asarray(train_acc)
len_data = myarray.shape[0]
myarray = np.reshape(myarray,(len_data,1))
train_array = np.reshape(train_array,(len_data,1))
myarray = [float(i[0]) for i in myarray]
train_array = [float(i[0]) for i in train_array]


plt.plot(np.arange(len_data),train_array[0:len_data],'b-',np.arange(len_data),myarray[0:len_data],'r')
plt.show()
