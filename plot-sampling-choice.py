"""
Created on Fri May  4 00:33:14 2018

@author: pydi
"""

import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.normal(-1,1,50)
x2 = np.random.normal(1,1,50)

lab1 = np.random.choice(x1,5) # Negative points
lab2 = np.random.choice(x2,5) # Positive points


w = -1
eta = 0.1
for _ in range(10):
        for item in lab1:
	        if item*w*(-1) < 0:
		        w = w + eta*item*(-1)

        for item in lab2:
	        if item*w*(1) < 0:
		        w = w + item*eta*(1)

r = np.arange(-7,7,0.01)
g1 = .05*(1/np.sqrt(2*np.pi))*np.exp(-(r+1)**2/2)
g2 = .05*(1/np.sqrt(2*np.pi))*np.exp(-(r-1)**2/2)

plt.figure(111)
x1_, = plt.plot(x1, np.zeros(x1.shape), 'rx', markersize=10)
x2_, = plt.plot(x2, np.zeros(x1.shape), 'bx', markersize=10)
lab1_, = plt.plot(lab1, np.zeros(lab1.shape), 'r8', markersize=8)
lab2_, = plt.plot(lab2, np.zeros(lab2.shape), 'b8', markersize=8)
g1_, = plt.plot(r, g1, 'r--', label = '- pt density')
g2_, = plt.plot(r, g2, 'b--', label = '+ pt density')
plt.axvline(x = w, color='k', label= 'trained classifier')
plt.text(w, 0.02, 'trained classifier', rotation=90)

l1 = 1+np.abs(w*x1)
l2 = 1+np.abs(w*x2)

xall = np.concatenate((x1,x2))
lall = np.concatenate((l1,l2))
lall = lall/np.sum(lall)

#plt.plot(xall, lall, 'g*')

for j in range(len(xall)):
        x = xall[j]
	loss = lall[j]
        g_, = plt.plot([xall[j], xall[j]], [0.0005, lall[j]], 'g-', label='sampling distribution')

plt.ylabel('Probability Distribution')
plt.xlabel('Data items')
oplt.legend(handles=[g_, g1_, g2_])

plt.show()
