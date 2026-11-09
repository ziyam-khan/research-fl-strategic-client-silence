import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def dirichlet_split(dataset, num_clients, alpha, seed=42):
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(targets == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        splits = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        for i, chunk in enumerate(np.split(class_idx, splits)):
            client_indices[i].extend(chunk.tolist())

    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return client_indices


def get_mnist(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train, test


def get_emnist(data_dir="./data", split="balanced", download=True):
    """EMNIST-balanced: 47 classes, 112,800 train. FEMNIST-style character task.

    Note: EMNIST images are transposed relative to MNIST. We do not correct this —
    train and test share the orientation, so classification is unaffected.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.EMNIST(data_dir, split=split, train=True, download=download, transform=transform)
    test = datasets.EMNIST(data_dir, split=split, train=False, download=download, transform=transform)
    return train, test


def get_cifar10(data_dir="./data", cls=datasets.CIFAR10, download=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = cls(data_dir, train=True, download=download, transform=transform_train)
    test = cls(data_dir, train=False, download=download, transform=transform_test)
    return train, test


def get_cifar100(data_dir="./data", download=True):
    """CIFAR-100 with CIFAR-10's augmentation and normalisation (PREREG-v2: isolates class count)."""
    return get_cifar10(data_dir, datasets.CIFAR100, download)


class HAR(torch.utils.data.Dataset):
    """UCI HAR (smartphone activity, 6 classes, 561 features already scaled to [-1, 1]).
    Expects data_dir/UCI HAR Dataset/{train,test}/{X,y,subject}_{train,test}.txt."""
    def __init__(self, data_dir, train=True):
        part = "train" if train else "test"
        d = os.path.join(data_dir, "UCI HAR Dataset", part)
        self.data = torch.from_numpy(np.loadtxt(os.path.join(d, f"X_{part}.txt"), dtype=np.float32))
        self.targets = (np.loadtxt(os.path.join(d, f"y_{part}.txt"), dtype=np.int64) - 1).tolist()
        self.subjects = np.loadtxt(os.path.join(d, f"subject_{part}.txt"), dtype=np.int64).tolist()
        assert self.data.shape == (len(self.targets), 561) and set(self.targets) == set(range(6))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]


def get_har(data_dir="./data", download=False):  # never downloads; setup_v2.sh fetches it
    return HAR(data_dir, train=True), HAR(data_dir, train=False)


def subject_split(dataset):
    """Natural federation: one client per subject (21 in the HAR train split), in subject order."""
    subj = np.array(dataset.subjects)
    return [np.where(subj == s)[0].tolist() for s in sorted(set(subj.tolist()))]


def make_client_loaders(dataset, client_indices, batch_size=32):
    loaders = []
    for indices in client_indices:
        subset = Subset(dataset, indices)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
    return loaders


def get_client_class_distribution(dataset, client_indices, num_classes=10):
    targets = np.array(dataset.targets)
    distributions = []
    for indices in client_indices:
        classes, counts = np.unique(targets[indices], return_counts=True)
        dist = np.zeros(num_classes)
        dist[classes] = counts / counts.sum()
        distributions.append(dist)
    return np.array(distributions)
