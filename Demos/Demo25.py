import numpy as np
import matplotlib.pyplot as plt

print("Demo 25: Distribucion Normal o Gaussiana en NumPy")
media = 130
ds = 50
cantidad = 1000*1000
normal = np.random.normal(media, ds, cantidad)
plt.hist(normal, 256)
plt.show()