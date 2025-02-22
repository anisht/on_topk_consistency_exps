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


PRINT = False
def generate_dataset(train_samples=100, test_samples=50, N=5, k=3, alpha=1, scale=1):
    parameter = np.array([.15, .15, .45, .25])*scale
    # parameter = [alpha]*N

    # train_data = np.random.dirichlet(parameter, size=train_samples)
    # sample each distribution many times
    # train_data = np.repeat(np.random.dirichlet(parameter, size=int(train_samples/1000)), 1000, axis=0)
    # distributions = {
    #     (.15, .15, .45, .25): [.15, .15, .45, .25], 
    #     (.15, .45, .15, .25): [.15, .45, .15, .25], 
    #     (.45, .15, .15, .25): [.45, .15, .15, .25], 
    #     # [.33, .33, .09, .25],
    # }
    # train_data = np.repeat([[.15, .15, .45, .25], [.15, .45, .15, .25], [.45, .15, .15, .25]], 1000, axis=0)
    # test_data = np.random.dirichlet(parameter, size=test_samples)
    # test_data = np.repeat([[.15, .15, .45, .25], [.15, .45, .15, .25], [.45, .15, .15, .25]], 1000, axis=0)

    train_data = np.array([np.random.dirichlet(parameter) for i in range(train_samples)])
    test_data = np.array([np.random.dirichlet(parameter) for i in range(test_samples)])

    

    # p = 0.3
    # q = 0.5 - p
    # points_list = [
    #     [p, q, .25, .25],
    #     [p, .25, q, .25],
    #     [p, .25, .25, q],
    #     [q, p, .25, .25],
    #     [.25, p, q, .25],
    #     [.25, p, .25, q],
    #     [q, .25, p, .25],
    #     [.25, q, p, .25],
    #     [.25, .25, p, q],
    #     [q, .25, .25, p],
    #     [.25, q, .25, p],
    #     [.25, .25, q, p],
    # ]
    # train_data = np.array(points_list * train_samples)
    # test_data = np.array(points_list * test_samples)


    # train_data = np.array([points_list[i] for i in np.random.choice(range(len(points_list)), size=train_samples)])
    # test_data = np.array([points_list[i] for i in np.random.choice(range(len(points_list)), size=test_samples)])


    # train_labels = np.array([[1, 1, 0] for i in range(train_samples)])
    # test_labels = np.array([[1, 1, 0]for i in range(test_samples)])

    # for i in range(train_samples):
    #     print(train_data[i])

    # train_topk = np.argsort(train_data, axis=1)[:,-k:]
    # test_topk = np.argsort(test_data, axis=1)[:,-k:]

    # train_labels = train_data.argmax(axis=1)
    # test_labels = test_data.argmax(axis=1)
    # train_labels = np.array([np.random.choice(train_topk[i]) for i in range(len(train_topk))])
    # test_labels = np.array([np.random.choice(test_topk[i]) for i in range(len(test_topk))])
    train_labels = np.array([np.random.choice(N, p=train_data[i]) for i in range(len(train_data))])
    test_labels = np.array([np.random.choice(N, p=test_data[i]) for i in range(len(test_data))])
    # train_labels = (train_labels + 1) % 4
    # test_labels = (test_labels + 1) % 4

    # total = 0 
    # for i in range(len(train_data)):
    #     if train_data[i][train_labels[i]] == q:
    #         total += 1
    # print("q samples:  ", total / len(train_data) )
    # train_point_count = (train_data[train_labels] == q).sum() / len(train_data)
    # test_point_count = sum([i[1] == 0.25 for i in test_data]) / len(test_data)
    # print("train dist: ", train_point_count, "\t test dist: ", test_point_count)
    # _, train_label_counts = np.unique(train_labels, return_counts=True)
    # _, test_label_counts = np.unique(test_labels, return_counts=True)
    # print("train labels: ", train_label_counts / len(train_data), "\t test labels: ", test_label_counts / len(test_data))


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
    net = nn.Linear(d, M, bias=False)
    eta = 0.01
    # net.weight.data.copy_(torch.eye(d))
    # net = nn.Sequential(
    #     nn.Linear(d, 20),
    #     nn.Sigmoid(),
    #     nn.Linear(20, M)
    # )
    optim = torch.optim.Adam(net.parameters(), eta)
    train_dataset = Dset(train_data, train_labels)
    if batch_size==None:
        batch_size=n
    train_dataloader = tdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    naive_test_s = torch.Tensor(test_data).detach().numpy()
    naive_tk_acc = 0
    naive_argsort_s = np.argsort(naive_test_s, axis=1)
    naive_test_n = naive_test_s.shape[0]
    for i in range(k):
        naive_tk_acc += sum(naive_argsort_s[:, -i-1] == test_labels)/naive_test_n
    if PRINT: print("IDENTITY TOPK LOSS: ", 1 - naive_tk_acc)

    for epoch in range(EPOCHS):
        if epoch % (EPOCHS / 10) == 0:
            test_s = net(torch.Tensor(test_data)).detach().numpy()
            testloss = loss_fn(torch.Tensor(test_s), torch.LongTensor(test_labels)).item()
            test_n = test_s.shape[0]

            tk_acc = 0
            argsort_s = np.argsort(test_s, axis=1)
            for i in range(k):
                tk_acc += sum(argsort_s[:, -i-1] == test_labels)/test_n

            if PRINT:
                print("TEST LOSS AT EPOCH ", epoch, ": ", testloss, "\tTOPK LOSS: ", 1 - tk_acc)
        for x,y in train_dataloader:
            optim.zero_grad()
            s = net(x)
            loss = loss_fn(s, y)
            loss.backward()
            optim.step()
           
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

    train_s = net(torch.Tensor(train_data)).detach().numpy()
    trainloss = loss_fn(torch.Tensor(train_s), torch.LongTensor(train_labels)).item()
    if PRINT: print("TRAIN LOSS: ", trainloss)

        
    return net, loss, acc, 1 - tk_acc


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
                     batch_size=None, scale=1):
    # N,k,l,M,d,c,f=NklMdcf
    # if K == None:
    #     K=5
    # 1st dim: loss type, 2nd dim: statistic type, 3/4 dim: trial1/trial2 number
    train_samples, test_samples = 10000, 1000

    result = np.zeros((len(loss_dict), 3, n_trials1))
    for i in range(n_trials1):
        train_data, train_labels, test_data, test_labels = generate_dataset(train_samples, test_samples, N=N, k=K, alpha=alpha, scale=scale)
        # print(NklMdcf)
        # print(train_data.shape)
        # print(train_labels.shape)
        # print(test_data.shape)
        # train_data, train_labels, test_data, test_labels, means = generate_gaussians2(N,k,l,M,d,c,f)
        perm = np.random.permutation(len(train_data))
        train_data, train_labels = train_data[perm], train_labels[perm]

        for m, (loss_name, loss_fn) in enumerate(loss_dict.items()):
            if PRINT: print("loss={}, i={}, scale={}".format(loss_name, i, scale))
            net, loss, acc, topk_acc = train_KM_and_evaluate(train_data, train_labels, test_data, 
                                                         test_labels, K, hidden_size = hidden_size, EPOCHS=EPOCHS, 
                                                         loss_fn=loss_fn, batch_size=batch_size)
            result[m, :, i] = [loss, acc, topk_acc]
            if PRINT:
                print(f"Loss: {loss}, acc: {acc}, top-{K} acc: {topk_acc}")
                for name, param in net.named_parameters():
                    print(name, ": ", param)
                print("-------------------------------------------------------------")
        if PRINT: print("================================================================")
    print("_____________________________________________________________________")
    return result