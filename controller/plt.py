import matplotlib.pyplot as plt
import numpy as np
f = open('e:/python/001/data/loss.txt','r')
data = f.readlines()[0]
loss = list(map(eval,data.split(',')[1:-1]))
x = range(0,len(loss)*100,100)
y = loss

plt.plot(x,y,color='red')
plt.grid()
plt.show()