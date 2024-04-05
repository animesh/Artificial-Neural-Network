#%%data
#!pip install numpy
import numpy as np
x=np.asarray([0.05,0.10])
y=np.asarray([0.01,0.99])
w1=np.asarray([[0.15,0.20],[0.25,0.3]])
w2=np.asarray([[0.4,0.45],[0.5,0.55]])
bias=[0.35,0.6]
lr=0.5
print("input",x,"\noutput",y,"\nbias",bias,"\nweights",w1,w2)
#%%forward-pass
h=1/(1+np.exp(-(x.dot(w1.T)+bias[0])))
y_pred=1/(1+np.exp(-(h.dot(w2.T)+bias[1])))
print("MSE",0.5*np.square(y_pred - y).sum())
#0.2983711087600027
#%%back-propagation
w3=w2-lr*np.outer((y_pred - y)*(1-y_pred)*y_pred,h)
print("weights updated",w3)
#[[0.35891648 0.40866619] [0.51130127 0.56137012]]
w4=w1-lr*np.outer(w2.T.dot((y_pred - y)*(1-y_pred)*y_pred)*h*(1-h),x)
print(w4)
#[[0.14978072 0.19956143] [0.24975114 0.29950229]]
h1=1/(1+np.exp(-(x.dot(w4.T)+bias[0])))
y_pred_h1=1/(1+np.exp(-(h1.dot(w3.T)+bias[1])))
print("MSE after iteration 1",0.5*np.square(y_pred_h1 - y).sum())
#0.29102777369359933
