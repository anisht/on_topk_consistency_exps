import numpy as np 
import torch
import torch.nn as nn 
import pandas as pd

from losses import *
from utils import repeat_experiment, repeat_experiment2

k=5
loss_dict = {'ent': nn.CrossEntropyLoss(), 'psi1':psi1(k), 'psi2':psi2(k), 'psi3':psi3(k),
            'psi4':psi4(k), 'psi5':psi5(k).apply, 'enta':trent1(k), 'entb':trent2(k)}

k=4
loss_dict2 = {'ent': nn.CrossEntropyLoss(), 'psi1':psi1(k), 'psi2':psi2(k), 'psi3':psi3(k),
            'psi4':psi4(k), 'psi5':psi5(k).apply, 'enta':trent1(k), 'entb':trent2(k)}

if __name__ == '__main__':
	exp_num = int(input('Choose which experiment (enter 1 or 2): '))
	if exp_num == 1:
		NklMdcf = [10, 5, 40, 10, 2, 5, 10]
		exp_res = []
		for N in [10, 50, 100]:
			NklMdcf[0] = N
			print(f"Running 10 trials for N={N}, exp={exp_num}")
			exp_res.append(repeat_experiment2(NklMdcf, loss_dict, 10, EPOCHS=500))
		print(f"Running 10 trials for N={N}, loss function k={4}, exp={exp_num}")
		exp_res.append(repeat_experiment2(NklMdcf, loss_dict2, 10, EPOCHS=500))

	if exp_num == 2:
		Nkldcf = [10, 5, 20, 2, 2, 1]
		exp_res = []
		for N in [10, 50, 100]:
			print(f"Running 10 trials for N={N}, exp={exp_num}")
			exp_res.append(repeat_experiment(Nkldcf, loss_dict, 10, EPOCHS=500))

	with open(f"exp_{exp_num}.pkl", 'wb') as f:
		pickle.dump(exp_res, f)

	for i, N in enumerate([10, 50, 100]):
		print(f"results for N={N} averaged over 10 trials:")
		res = exp_res[i]
		df = pd.DataFrame(res.mean(axis=2), index=loss_dict.keys(), columns=['loss', 'acc', 'top-5'])
		print(df)
	if exp_num == 1:
		print(f"results for N={N}, loss function k={4} averaged over 10 trials:")
		res = exp_res[-1]
		df = pd.DataFrame(res.mean(axis=2), index=loss_dict.keys(), columns=['loss', 'acc', 'top-5'])
		print(df)


