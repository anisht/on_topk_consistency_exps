import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
# activate latex text rendering
rc('text', usetex=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
plt.rc('font', **font)
# plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots()
ax.set_xscale('log', base=2)

xs = [128,256,512,1024,2048]
L2 = [0.643881043480206, 0.7135363330879873, 0.6968910948573259, 0.7254232487144938, 0.711903968158073]
L3 = [0.649781, 0.71994, 0.708584, 0.732353, 0.720233]
L4 = [0.595903, 0.69845, 0.673394, 0.70917, 0.694772]
Lk = [0.201426, 0.244845, 0.252417, 0.257656, 0.261006]

ax.set_xticks(xs)
ax.set_xticklabels(xs)
plt.plot(xs, L2, label=r"$L^{(2)}$", linestyle='-.')
plt.plot(xs, L3, label=r"$L^{(3)}$", linestyle='--')
plt.plot(xs, L4, label=r"$L^{(4)}$", linestyle=':')
plt.plot(xs, Lk, label=r"$L_k$")
plt.xlabel(r"\# samples")
plt.ylabel("empirical loss")
plt.legend()
plt.savefig('ERM-plot-textured.png')
plt.show()

