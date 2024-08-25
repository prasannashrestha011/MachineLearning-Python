import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(0,10,6)
y=np.linspace(0,10,6)
plt.bar(x,y,width=1.0,color="blue",edgecolor="yellow",tick_label=x)
plt.plot(x,y,marker="x",linestyle="none",color="blue")
plt.show();