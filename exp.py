import numpy as np 
import torch
import torch.nn as nn 
import pandas as pd
import pickle

from losses import *
from utils import repeat_experiment, repeat_experiment2, repeat_experiment3


N = 4
k = 2

loss_dict = {
			 'L6':psi6(k),
			 'L2':psi2(k),
			 'L3':psi3(k),
			 'L4':psi4(k),
			 # 'ent': nn.CrossEntropyLoss(),
			 # 'L1':psi1(k),
			 # 'L5':psi5(k).apply,
			 # 'enta':trent1(k),
			 # 'entb':trent2(k)
			 }

# loss_dict2 = {
# 			  'L6':psi6(k),
# 			  'L2':psi2(k),
# 			  'L3':psi3(k),
# 			  'L4':psi4(k),
# 			  'ent': nn.CrossEntropyLoss(),
# 			  # 'L1':psi1(k),
# 			  # 'L5':psi5(k).apply,
# 			  # 'enta':trent1(k),
# 			  # 'entb':trent2(k)
# 			  }

if __name__ == '__main__':
	EPOCHS=200
	num_trials=5
	# Ns = [10, 50, 100]
	# exp_num = int(input('Choose which experiment (enter 1 or 2): '))
	exp_num = 1
	# if exp_num == 1:
	# NklMdcf = [10, 5, 40, 10, 2, 5, 10]
	alphas = [1]
	alpha = 1
	# alphas = [0.01,0.05,0.1,0.2,0.4,0.8]
	# alphas = [0.25,0.5,1,2,4,8]
	Ns=[N]

	alpha_res = pd.DataFrame()
	# for alpha in alphas:
	for scale in np.logspace(-1, 5, num=7):
		exp_res = []
		# for N in [10, 50, 100]:
		# for N in Ns:
		print(f"Running {num_trials} trials for N={N}, k={k}, exp={exp_num}, scale={scale}")
		exp_res.append(repeat_experiment3(loss_dict, num_trials, K=k, N=N, alpha=alpha, EPOCHS=EPOCHS, scale=scale))
		# print(f"Running 10 trials for N={N}, loss function k={4}, exp={exp_num}")
		# exp_res.append(repeat_experiment3(NklMdcf, loss_dict2, 10, EPOCHS=EPOCHS))

		# if exp_num == 2:
		# 	Nkldcf = [10, 5, 20, 2, 2, 1]
		# 	exp_res = []
		# 	for N in Ns:
		# 		print(f"Running 10 trials for N={N}, exp={exp_num}")
		# 		exp_res.append(repeat_experiment(Nkldcf, loss_dict, 10, EPOCHS=EPOCHS))

		# with open(f"exp_{exp_num}_EPOCHS_{EPOCHS}.pkl", 'wb') as f:
		# with open(f"max_label.pkl", 'wb') as f:
		# 	pickle.dump(exp_res, f)

		for i, N in enumerate(Ns):
			print(f"results for N={N} averaged over {num_trials} trials:")
			res = exp_res[i]
			df = pd.DataFrame(res.mean(axis=2), index=loss_dict.keys(), columns=['loss', 'acc', f'top-{k}'])
			print(df)
			alpha_res[alpha] = df[f'top-{k}']
		# if exp_num == 1:
		# 	print(f"results for N={N}, loss function k={4} averaged over 10 trials:")
		# 	res = exp_res[-1]
		# 	df = pd.DataFrame(res.mean(axis=2), index=loss_dict.keys(), columns=['loss', 'acc', 'top-5'])
		# 	print(df)
	# with open(f"alpha_plot_n{N}_k{k}_epochs{EPOCHS}.pkl", 'wb') as f:
	# 	pickle.dump(alpha_res, f)
	# print(alpha_res)

	# import matplotlib.pyplot as plt
	# import seaborn as sns
	# alpha_res.T.plot.line(logx=True, xticks=alphas)
	# plt.xlabel("alpha")
	# plt.ylabel("top-k accuracy")
	# plt.show()

