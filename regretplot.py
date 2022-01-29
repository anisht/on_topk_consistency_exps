import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
# activate latex text rendering
rc('text', usetex=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)
# plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots()
ax.set_xscale('log', base=2)

xs = [0.125,0.25,0.5,1,2,4,8]
L2 = [0.0150834, 0.0237575, 0.0384554, 0.0548851, 0.0612849, 0.0472543, 0.0275879]

# runs from sarangs computer
# L3a = [0.00499287, 0.00763982,0.0281417,0.0469146,0.0532453,0.0443298, 0.0261377]
# L3b = [0.00429697, 0.011327, 0.0202692, 0.0466014, 0.0649702,0.0462929, 0.0229119]
# L3 = [(L3a[i] + L3b[i])/2 for i in range(len(L3a))]

L3 = [0.00406396, 0.00897751, 0.0255483, 0.0461951, 0.0584822,0.0447333, 0.0245767]

L4 = [0.000613423, 0.00241527, 0.00716833, 0.0141337, 0.0165687, 0.00919684, 0.0044324]
Lk = [0, 0, 0, 0, 0, 0, 0]

ax.set_xticks(xs)
ax.set_xticklabels(xs)
plt.plot(xs, L2, label=r"$L^{(2)}$", linestyle='-.')
plt.plot(xs, L3, label=r"$L^{(3)}$", linestyle='--')
plt.plot(xs, L4, label=r"$L^{(4)}$", linestyle=':')
plt.plot(xs, Lk, label=r"$L_k$")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"Regret")
plt.legend()
plt.savefig('regret-plot-textured.png')
plt.show()

