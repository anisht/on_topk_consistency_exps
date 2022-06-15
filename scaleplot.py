data = {
    0.1: {"L6":0.0098, "L2": 0.0038, "L3":0.0032, "L4":0.0040},
    1: {"L6":0.1020, "L2": 0.0604, "L3":0.0598, "L4":0.0692},
    10: {"L6":0.2736, "L2": 0.2520, "L3":0.2478, "L4":0.2594},
    100: {"L6":0.3042, "L2": 0.3120, "L3":0.3204, "L4":0.3134},
    1000: {"L6":0.3002, "L2": 0.3346, "L3":0.3312, "L4":0.3328},
    10000: {"L6":0.3006, "L2": 0.3296, "L3":0.3442, "L4":0.3394},
    100000: {"L6":0.3028, "L2": 0.3368, "L3":0.3496, "L4":0.3414}
}

loss_data = {
    "L2":[],
    "L3":[],
    "L4":[],
    "L6":[],
}
for scale in data:
    for loss in data[scale]:
        loss_data[loss] += [data[scale][loss]]



import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import numpy as np
# activate latex text rendering
rc('text', usetex=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
plt.rc('font', **font)
# plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots()
ax.set_xscale('log', base=10)

xs = np.logspace(-1, 5, num=7)
L2 = loss_data["L2"]
L3 = loss_data["L3"]
L4 = loss_data["L4"]
Lk = loss_data["L6"]

ax.set_xticks(xs)
ax.set_xticklabels(xs)
plt.plot(xs, L2, label=r"$L^{(2)}$", linestyle='-.')
plt.plot(xs, L3, label=r"$L^{(3)}$", linestyle='--')
plt.plot(xs, L4, label=r"$L^{(4)}$", linestyle=':')
plt.plot(xs, Lk, label=r"$L_k$")
plt.xlabel(r"alpha")
plt.ylabel("top-2 loss")
plt.legend()
plt.savefig('scale-accuracy-plot.png')
plt.show()

