import numpy as np 
import gc
import torch
import torch.nn as nn 
import torch.utils.data as tdata

def generate_gaussians(N=5, k=3, l=10, d=None, c=5, f=100):
    """Generates N*k*l samples from gaussian distributions. First computes N Gaussian means
    (which are made to be spaced sufficiently far apart), and then draws k*l samples from each 
    mean. Each mean is given k different labels (to motivate top-k), and l points drawn from 
    the mean are assigned to each label. 
    ==================================================================
    N: number of means
    k: number of classes per mean
    l: number of points per class
    d: dimension of gaussian samples
    c: how many standard deviations apart the gaussian centers have to be
    f: how many times higher the variance of the gaussian used to draw the centers is than 
       the variance of each individual gaussian center divided by N
    """
    if d==None:
        d=N
    means = np.random.multivariate_normal(np.zeros(d), np.eye(d)).reshape((1,-1))
    for i in range(N-1):
        sample = np.random.multivariate_normal(np.zeros(d), N*f*np.eye(d))
        # make sure the means are at least c*sqrt(d) apart
        while np.linalg.norm(means-sample, axis=1).min() < c*np.sqrt(d):
            sample = np.random.multivariate_normal(np.zeros(d), N*f*np.eye(d))
        means = np.vstack((means, sample))
    
    gaussian_list = np.empty((0, d))
    test_list = np.empty((0, d))
    
    for i in range(l):
        for mean in means:
            samples = np.random.multivariate_normal(mean, np.eye(d), k)
            gaussian_list = np.vstack((gaussian_list, samples))
            if i%3==0:
                test_sample = np.random.multivariate_normal(mean, np.eye(d), k)
                test_list = np.vstack((test_list, test_sample))
        
    train_data, train_labels =  gaussian_list, np.tile(np.arange(N*k), l)
    test_data, test_labels  = test_list, np.tile(np.arange(N*k), int(np.ceil(l/3)))
    return train_data, train_labels, test_data, test_labels, means


def generate_gaussians2(N=5, k=3, l=10, M=20, d=None, c=5, f=100):
    """Generates N*k*l samples from gaussian distributions. First computes N Gaussian means
    (which are made to be spaced sufficiently far apart), and then draws k*l samples from each 
    mean. Each mean is given k different labels (to motivate top-k), and l points drawn from 
    the mean are assigned to each label. 
    ==================================================================
    N: number of means
    k: number of means per class
    l: number of points per class
    M: number of classes
    d: dimension of gaussian samples
    c: how many standard deviations apart the gaussian centers have to be
    f: how many times higher the variance of the gaussian used to draw the centers is than 
       the variance of each individual gaussian center divided by N
    """
    if d==None:
        d=N
    means = np.random.multivariate_normal(np.zeros(d), np.eye(d)).reshape((1,-1))
    for i in range(N-1):
        sample = np.random.multivariate_normal(np.zeros(d), N*f*np.eye(d))
        # make sure the means are at least c*sqrt(d) apart
        while np.linalg.norm(means-sample, axis=1).min() < c*np.sqrt(d):
            sample = np.random.multivariate_normal(np.zeros(d), N*f*np.eye(d))
        means = np.vstack((means, sample))
    
    mean_probs = np.random.rand(M,k)
    mean_probs = mean_probs/mean_probs.sum(axis=1).reshape(-1,1)
    
    class_means = np.array([np.random.choice(np.arange(N),k, replace=True) for i in range(M)])
    
    gaussian_list = np.empty((0, d))
    test_list = np.empty((0, d))
    
    tl = int(np.ceil(l/3))
    
    for i in range(M):
        mean_choices = np.random.choice(class_means[i], l, p=mean_probs[i])
        gaussian_list = np.vstack((gaussian_list, 
                            [np.random.multivariate_normal(means[choice], np.eye(d)) for choice in mean_choices]))
        
        mean_choices = np.random.choice(class_means[i], tl, p=mean_probs[i])
        test_list = np.vstack((test_list, 
                            [np.random.multivariate_normal(means[choice], np.eye(d)) for choice in mean_choices]))
        
    train_data, train_labels =  gaussian_list, np.tile(np.arange(M), l).reshape(l, M).T.flatten()
    test_data, test_labels  = test_list,  np.tile(np.arange(M), tl).reshape(tl, M).T.flatten()
    return train_data, train_labels, test_data, test_labels, mean_probs


def generate_dataset(train_samples=100, test_samples=50, N=4, k=2, alpha=1):
    parameter = [5, 2, 2]
    # parameter = [alpha]*N

    train_data = np.random.dirichlet(parameter, size=train_samples)
    test_data = np.random.dirichlet(parameter, size=test_samples)

    # for i in range(train_samples):
    #     print(train_data[i])

    # train_topk = np.argsort(train_data, axis=1)[:,-k:]
    # test_topk = np.argsort(test_data, axis=1)[:,-k:]

    # train_labels = train_data.argmax(axis=1)
    # test_labels = test_data.argmax(axis=1)
    # train_labels = np.array([np.random.choice(train_topk[i]) for i in range(len(train_topk))])
    # test_labels = np.array([np.random.choice(test_topk[i]) for i in range(len(test_topk))])
    train_labels = np.array([np.random.choice(N, p=train_data[i]) for i in range(train_samples)])
    test_labels = np.array([np.random.choice(N, p=test_data[i]) for i in range(test_samples)])

    return train_data, train_labels, test_data, test_labels

class Dset(tdata.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.Tensor(data)
        self.labels = torch.Tensor(labels).to(torch.long)
        self.transform = None

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx])

    def __len__(self):
        return len(self.labels)

def train_KM_and_evaluate(train_data, train_labels, test_data, test_labels, k, loss_fn,
                         hidden_size=64, EPOCHS=100, batch_size=None):
    n,d = train_data.shape
    M = np.max(train_labels)+1
    net = nn.Linear(d, M)
    # net = nn.Sequential(
    #     nn.Linear(d, 20),
    #     nn.Sigmoid(),
    #     nn.Linear(20, M)
    # )
    optim = torch.optim.Adam(net.parameters(), 0.1)
    train_data = Dset(train_data, train_labels)
    if batch_size==None:
        batch_size=n
    train_dataloader = tdata.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    for epoch in range(EPOCHS):
        for x,y in train_dataloader:
            s = net(x)
            loss = loss_fn(s, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
           
    # evaluate
    test_s = net(torch.Tensor(test_data)).detach().numpy()
    test_n = test_s.shape[0]
    # top 1
    acc = sum(np.argmax(test_s, axis=1) == test_labels)/test_n
    # top k 
    tk_acc = 0
    argsort_s = np.argsort(test_s, axis=1)
    for i in range(k):

        tk_acc += sum(argsort_s[:, -i-1] == test_labels)/test_n
    loss = loss_fn(torch.Tensor(test_s), torch.LongTensor(test_labels)).item()
        
    return net, loss, acc, tk_acc


def repeat_experiment(Nkldcf, loss_dict, n_trials1, K=None, hidden_size = 64, EPOCHS = 100,
                     batch_size=None):
    N,k,l,d,c,f=Nkldcf
    if K == None:
        K=k
    # 1st dim: loss type, 2nd dim: statistic type, 3/4 dim: trial1/trial2 number
    results = np.zeros((len(loss_dict), 3, n_trials1))
    for i in range(n_trials1):
        train_data, train_labels, test_data, test_labels, means = generate_gaussians(N,k,l,d,c,f)
        perm = np.random.permutation(len(train_data))
        train_data, train_labels = train_data[perm], train_labels[perm]
        gc.collect()
        for m, (loss_name, loss_fn) in enumerate(loss_dict.items()):
            print("loss={}, i={}, k={}".format(loss_name, i, k))
            net, loss, acc, topk_acc = train_KM_and_evaluate(train_data, train_labels, test_data, 
                                                         test_labels, K, hidden_size = hidden_size, EPOCHS=EPOCHS, 
                                                         loss_fn=loss_fn, batch_size=batch_size)
            results[m, :, i] = [loss, acc, topk_acc]
            print(f"Loss: {loss}, acc: {acc}, top-{k} acc: {topk_acc}")
            print("-------------------------------------------------------------")
        print("================================================================")
    print("_____________________________________________________________________")
    return results


def repeat_experiment2(NklMdcf, loss_dict, n_trials1, K=None, hidden_size = 64, EPOCHS = 100,
                     batch_size=None):
    N,k,l,M,d,c,f=NklMdcf
    if K == None:
        K=k
    # 1st dim: loss type, 2nd dim: statistic type, 3/4 dim: trial1/trial2 number
    results = np.zeros((len(loss_dict), 3, n_trials1))
    for i in range(n_trials1):
        train_data, train_labels, test_data, test_labels, means = generate_gaussians2(N,k,l,M,d,c,f)
        # print(NklMdcf)
        # print(train_data.shape)
        # print(train_labels.shape)
        # print(test_data.shape)
        train_data, train_labels, test_data, test_labels, means = generate_gaussians2(N,k,l,M,d,c,f)
        perm = np.random.permutation(len(train_data))
        train_data, train_labels = train_data[perm], train_labels[perm]

        for m, (loss_name, loss_fn) in enumerate(loss_dict.items()):
            print("loss={}, i={}, k={}".format(loss_name, i, K))
            net, loss, acc, topk_acc = train_KM_and_evaluate(train_data, train_labels, test_data, 
                                                         test_labels, K, hidden_size = hidden_size, EPOCHS=EPOCHS, 
                                                         loss_fn=loss_fn, batch_size=batch_size)
            results[m, :, i] = [loss, acc, topk_acc]
            print(f"Loss: {loss}, acc: {acc}, top-{K} acc: {topk_acc}")
            print("-------------------------------------------------------------")
        print("================================================================")
    print("_____________________________________________________________________")
    return results

def repeat_experiment3(loss_dict, n_trials1, K=2, N=4, alpha=1, hidden_size = 64, EPOCHS = 100,
                     batch_size=None):
    # N,k,l,M,d,c,f=NklMdcf
    # if K == None:
    #     K=5
    # 1st dim: loss type, 2nd dim: statistic type, 3/4 dim: trial1/trial2 number
    train_samples, test_samples = 5000, 1000

    results = np.zeros((len(loss_dict), 3, n_trials1))
    for i in range(n_trials1):
        train_data, train_labels, test_data, test_labels = generate_dataset(train_samples, test_samples, N=N, k=K, alpha=alpha)
        # print(NklMdcf)
        # print(train_data.shape)
        # print(train_labels.shape)
        # print(test_data.shape)
        # train_data, train_labels, test_data, test_labels, means = generate_gaussians2(N,k,l,M,d,c,f)
        perm = np.random.permutation(len(train_data))
        train_data, train_labels = train_data[perm], train_labels[perm]

        for m, (loss_name, loss_fn) in enumerate(loss_dict.items()):
            print("loss={}, i={}, k={}".format(loss_name, i, K))
            net, loss, acc, topk_acc = train_KM_and_evaluate(train_data, train_labels, test_data, 
                                                         test_labels, K, hidden_size = hidden_size, EPOCHS=EPOCHS, 
                                                         loss_fn=loss_fn, batch_size=batch_size)
            results[m, :, i] = [loss, acc, topk_acc]
            print(f"Loss: {loss}, acc: {acc}, top-{K} acc: {topk_acc}")
            # for name, param in net.named_parameters():
            #     print(name, ": ", param)
            print("-------------------------------------------------------------")
        print("================================================================")
    print("_____________________________________________________________________")
    return results