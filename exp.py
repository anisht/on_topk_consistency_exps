import numpy as np 
import torch
import torch.nn as nn 
import pandas as pd
import pickle

from losses import *
from utils import repeat_experiment, repeat_experiment2, repeat_experiment3

k=5
loss_dict = {'ent': nn.CrossEntropyLoss(),
			 'L1':psi1(k),
			 'L2':psi2(k),
			 'L3':psi3(k),
			 'L4':psi4(k),
			 'L5':psi5(k).apply,
			 'L6':psi6(k),
			 'enta':trent1(k),
			 'entb':trent2(k)
			 }

k=4
loss_dict2 = {'ent': nn.CrossEntropyLoss(),
			  'L1':psi1(k),
			  'L2':psi2(k),
			  'L3':psi3(k),
			  'L4':psi4(k),
			  'L5':psi5(k).apply,
			  'L6':psi6(k),
			  'enta':trent1(k),
			  'entb':trent2(k)
			  }

if __name__ == '__main__':
	EPOCHS=100
	num_trials=2
	# Ns = [10, 50, 100]
	# exp_num = int(input('Choose which experiment (enter 1 or 2): '))
	exp_num = 1
	# if exp_num == 1:
	# NklMdcf = [10, 5, 40, 10, 2, 5, 10]
	k = 5
	N = 10
	Ns=[N]
	exp_res = []
	# for N in [10, 50, 100]:
	# for N in Ns:
	print(f"Running 10 trials for N={N}, exp={exp_num}")
	exp_res.append(repeat_experiment3(loss_dict, num_trials, K=k, N=N, EPOCHS=EPOCHS))
	# print(f"Running 10 trials for N={N}, loss function k={4}, exp={exp_num}")
	# exp_res.append(repeat_experiment3(NklMdcf, loss_dict2, 10, EPOCHS=EPOCHS))

	# if exp_num == 2:
	# 	Nkldcf = [10, 5, 20, 2, 2, 1]
	# 	exp_res = []
	# 	for N in Ns:
	# 		print(f"Running 10 trials for N={N}, exp={exp_num}")
	# 		exp_res.append(repeat_experiment(Nkldcf, loss_dict, 10, EPOCHS=EPOCHS))

	# with open(f"exp_{exp_num}_EPOCHS_{EPOCHS}.pkl", 'wb') as f:
	with open(f"max_label.pkl", 'wb') as f:
		pickle.dump(exp_res, f)

	for i, N in enumerate(Ns):
		print(f"results for N={N} averaged over 10 trials:")
		res = exp_res[i]
		df = pd.DataFrame(res.mean(axis=2), index=loss_dict.keys(), columns=['loss', 'acc', 'top-5'])
		print(df)
	# if exp_num == 1:
	# 	print(f"results for N={N}, loss function k={4} averaged over 10 trials:")
	# 	res = exp_res[-1]
	# 	df = pd.DataFrame(res.mean(axis=2), index=loss_dict.keys(), columns=['loss', 'acc', 'top-5'])
	# 	print(df)


