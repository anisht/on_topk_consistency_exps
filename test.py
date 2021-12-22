import numpy as np 
def generate_dataset(samples=100, n=10, k=2):
    test_data = np.random.rand(samples, n)
    train_data = np.random.rand(samples, n)

    test_topk = np.argsort(test_data, axis=1)[:,-k:]
    train_topk = np.argsort(train_data, axis=1)[:,-k:]

    # test_labels = test_data.argmax(axis=1)
    # train_labels = train_data.argmax(axis=1)
    test_labels = np.array([np.random.choice(test_topk[i]) for i in range(len(test_topk))])
    train_labels = np.array([np.random.choice(train_topk[i]) for i in range(len(train_topk))])

    return train_data, train_labels, test_data, test_labels

generate_dataset(5, 10, 2)