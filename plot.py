import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['text.usetex']=False


# ----------------------------------------------------------------------------


x = np.linspace(-6,2,100)
plt.figure(figsize=(15,5))
plt.plot(x, np.sign(x), label="0/1")
plt.plot(x, (1-x)**2, label="square")
plt.plot(x, np.maximum((1-x)**2,0), label="modified LS")
plt.plot(x, np.maximum((1-x),0), label="SVM")
plt.plot(x, np.exp(-x), label="Boost")
plt.plot(x, np.log(1+np.exp(-x)), label="Logistic")
plt.plot(x, 1/((1+np.exp(x))**2), label="Savage")
plt.plot(x, (2*np.arctan(x)-1)**2, label="Tangent")
plt.xlim(-1,2)
plt.ylim(-2, 4)
plt.title("Classification loss functions (y=1)")
plt.legend()
plt.savefig("plots/classification.svg")


# ----------------------------------------------------------------------------


x = np.linspace(-2,3,100)
out = (np.abs(1-x)-0.5)
out[np.where(np.abs(1-x)<1)] = 0.5*(1-x[np.where(np.abs(1-x)<1)])**2
plt.figure(figsize=(5,5))
plt.plot(x, (1-x)**2, label=r"square")
plt.plot(x, np.abs(1-x), label=r"absolute")
plt.plot(x, out, label="Huber, delta = 1")
plt.plot(x, np.log(np.cosh(1-x)), label=r"log-ch")
plt.xlim(-1,3)
plt.ylim(-0.5, 4.5)
plt.title(r"Regression loss functions (y=1)")
plt.legend()
plt.savefig("plots/reg.svg")
