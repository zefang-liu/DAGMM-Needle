import numpy as np
import pandas as pd
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizontally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return img[:, ::-1, :]
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of clipped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        H, W, C = img.shape
        img_padded = np.pad(img, (
            (self.padding, self.padding),
            (self.padding, self.padding),
            (0, 0)))
        x_start = self.padding + shift_x
        y_start = self.padding + shift_y
        return img_padded[x_start:(x_start + H), y_start:(y_start + W), :]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.batch_index = 0
        if self.shuffle:
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            self.ordering = np.array_split(
                indices,
                range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.batch_index < len(self.ordering):
            batch = {}

            for index in self.ordering[self.batch_index]:
                data = self.dataset[index]
                for i, _data in enumerate(data):
                    if i not in batch:
                        batch[i] = []
                    batch[i].append(_data)

            self.batch_index += 1
            return tuple(Tensor(np.array(batch[i])) for i in range(len(batch)))
        else:
            raise StopIteration
        ### END YOUR SOLUTION


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dtype=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    import gzip, struct

    with gzip.open(image_filesname, 'rb') as image_files:
        header = struct.unpack('>4I', image_files.read(16))
        magic, size, width, height = header
        X = np.frombuffer(
            image_files.read(size * width * height), dtype=np.uint8
        ).astype(np.float32).reshape((size, width * height)) / 255

    with gzip.open(label_filename, 'rb') as label_file:
        header = struct.unpack('>2I', label_file.read(8))
        magic, size = header
        y = np.frombuffer(
            label_file.read(size), dtype=np.uint8
        ).reshape((size))

    return X, y
    ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        X, y = parse_mnist(image_filename, label_filename)
        H = W = int(np.sqrt(X.shape[1]))
        self.X = X.reshape((-1, H, W, 1))
        self.y = y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        return self.apply_transforms(self.X[index]), self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.C, self.H, self.W = 3, 32, 32
        self.p = p
        self.transforms = transforms

        X = []
        y = []

        if train:
            for batch in range(1, 6):
                batch_file_path = os.path.join(
                    base_folder, f'data_batch_{batch}')

                with open(batch_file_path, 'rb') as batch_file:
                    batch_data = pickle.load(batch_file, encoding='bytes')
                    X.append(batch_data[b'data'])
                    y.extend(batch_data[b'labels'])
        else:
            batch_file_path = os.path.join(
                base_folder, 'test_batch')

            with open(batch_file_path, 'rb') as batch_file:
                batch_data = pickle.load(batch_file, encoding='bytes')
                X = batch_data[b'data']
                y = batch_data[b'labels']

        self.X = np.array(X).astype(np.float32) \
            .reshape((-1, self.C, self.H, self.W)) / 255
        self.y = y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        return self.apply_transforms(self.X[index]), self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
        return self.word2idx[word]
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.word2idx)
        ### END YOUR SOLUTION


class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        ids = []

        with open(path, 'r') as file:
            num_lines = 0
            while True:
                line = file.readline()
                num_lines += 1

                if (not line) or (max_lines and num_lines > max_lines):
                    break
                else:
                    tokens = line.split()
                    
                    for token in tokens:
                        ids.append(self.dictionary.add_word(token))
                    
                    ids.append(self.dictionary.add_word('<eos>'))

        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e.g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    nbatch = len(data) // batch_size
    data = data[:nbatch * batch_size]
    array = np.array(data, dtype=dtype).reshape(
        batch_size, nbatch).transpose()
    return array
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivision of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    seq_len, batch_size = batches.shape
    seq_len = min(bptt, seq_len - i - 1)
    data = Tensor(batches[i:i + seq_len, :], 
        device=device, dtype=dtype)
    target = Tensor(batches[i + 1:i + seq_len + 1, :].reshape(-1), 
        device=device, dtype=dtype)
    return data, target
    ### END YOUR SOLUTION


def parse_kdd_cup():
    """
    Load and preprocess the KDD Cup 1999 Dataset
    """
    ### BEGIN YOUR SOLUTION
    from sklearn.datasets import fetch_kddcup99
    data = fetch_kddcup99(shuffle=False, percent10=True, as_frame=True)

    df_kdd_cup = data.frame
    df_kdd_cup.loc[df_kdd_cup.labels != b'normal.', 'labels'] = 0
    df_kdd_cup.loc[df_kdd_cup.labels == b'normal.', 'labels'] = 1
    df_kdd_cup = df_kdd_cup.convert_dtypes()

    object_columns = df_kdd_cup.dtypes[df_kdd_cup.dtypes == 'object'].index
    one_hot_tables = []
    for object_column in object_columns:
        one_hot_tables.append(pd.get_dummies(df_kdd_cup[object_column]))

    df_numerics = pd.concat([*one_hot_tables, df_kdd_cup.drop(columns=object_columns)], axis=1)
    df_normal = df_numerics[df_numerics.labels == 0]
    df_abnormal = df_numerics[df_numerics.labels == 1]
    features = df_numerics.drop(columns='labels')
    features_normal = df_normal.drop(columns='labels')
    features_abnormal = df_abnormal.drop(columns='labels')
    labels_normal = df_normal.labels
    labels_abnormal = df_abnormal.labels

    X_normal = ((features_normal - features.min()) / (features.max() - features.min())).values.astype('float32')
    X_abnormal = ((features_abnormal - features.min()) / (features.max() - features.min())).values.astype('float32')
    y_normal = labels_normal.values.astype('uint8')
    y_abnormal = labels_abnormal.values.astype('uint8')
    return X_normal, X_abnormal, y_normal, y_abnormal
    ### END YOUR SOLUTION


class KDDCUPDataset(Dataset):
    """
    KDD Cup 1999 Dataset
    """
    def __init__(self, train=True, train_ratio=0.5):
        ### BEGIN YOUR SOLUTION
        super().__init__()

        from sklearn.model_selection import train_test_split
        X_normal, X_abnormal, y_normal, y_abnormal = parse_kdd_cup()
        assert 1 - train_ratio >= len(y_abnormal) / (len(y_normal) + len(y_abnormal))
        train_size = int(train_ratio * (len(y_normal) + len(y_abnormal)))

        X_train, X_normal_test, y_train, y_normal_test = train_test_split(
            X_normal, y_normal, train_size=train_size, shuffle=True, random_state=0)
        X_test = np.concatenate((X_normal_test, X_abnormal), axis=0)
        y_test = np.concatenate((y_normal_test, y_abnormal), axis=0)

        if train:
            self.X = X_train
            self.y = y_train
        else:
            self.X = X_test
            self.y = y_test
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        return self.X[index], self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION
