
from trainModel import Model
import numpy as np
import matplotlib.pyplot as plt
loss = []
accuracy = []
m = Model(
        'data/dl-data/couplet/train/in.txt',
        'data/dl-data/couplet/train/out.txt',
        'data/dl-data/couplet/test/in.txt',
        'data/dl-data/couplet/test/out.txt',
        'data/dl-data/couplet/vocabs',
        list_loss = loss,
        list_accuracy = accuracy,
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.001,
        output_dir='data/dl-data/models/tf-lib/output_couplet',
        restore_model=False)

m.train(200)
x1 = range(len(m.list_loss))
x2 = range(len(m.list_accuracy))
y1 = m.list_loss
y2 = m.list_accuracy
plt.figure(1)

#第一行第一列图形
ax1 = plt.subplot(2,1,1)
#第一行第二列图形
ax2 = plt.subplot(2,1,2)

plt.sca(ax1)
plt.plot(x1,y1,label="loss",color='red')
plt.xlabel("time")
plt.ylabel("loss")

plt.sca(ax2)
plt.plot(x2,y2,label="accuracy",color='green')
plt.xlabel("time")
plt.ylabel("accuracy")

plt.show()
