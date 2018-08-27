"""
    Time: 18-8-1 下午2:43
    Author: sharejing
    Description: 处理数据集的类即 Data processing/loading helpers

"""
import unicodedata
from torch.utils.data import Dataset
from .vector import vectorize
from torch.utils.data.sampler import Sampler
import numpy as np


# Dictionary class for tokens
class Dictionary(object):
    NULL = "<NULL>"
    UNK = "UNK"
    START = 2

    @staticmethod
    def normalize(token):
        """
        将Unicode文本标准化
        :param token:
        :return:
        """
        return unicodedata.normalize("NFD", token)

    def __init__(self):
        self.tok2ind = {self.NULL: 0, self.UNK: 1}
        self.ind2tok = {0: self.NULL, 1: self.NULL}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, item):
        if type(item) == int:
            return item in self.ind2tok
        elif type(item) == str:
            return self.normalize(item) in self.tok2ind

    def __getitem__(self, item):
        if type(item) == int:
            return self.ind2tok.get(item, self.UNK)
        if type(item) == str:
            return self.tok2ind.get(self.normalize(item),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, value):
        if type(key) == int and type(value) == str:
            self.ind2tok[key] = value
        elif type(key) == str and type(value) == int:
            self.tok2ind[key] = value
        else:
            raise RuntimeError("Invalid (key, value) types")

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """
        Get dict tokens
        :return: Return all the words indexed by this dict, except for UNK and NULL
        """
        tokens = [k for k in self.tok2ind.keys() if k not in ["<NULL>", "<UNK>"]]

        return tokens


# PyTorch dataset class for SQuAD(and SQuAD-like) data.
class ReaderDataset(Dataset):

    def __init__(self, examples, model, single_answer=False):
        self.examples = examples
        self.model = model
        self.single_answer = single_answer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return vectorize(self.examples[item], self.model, self.single_answer)

    def lengths(self):
        return [(len(ex["document"]), len(ex["question"]))for ex in self.examples]


# PyTorch sampler returning batched of sorted lengths(by doc and question)
class SortedBatchSampler(Sampler):

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):

        # lengths: (-len_doc, -len_question, 一个随机数)
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths],
            dtype=[("l1", np.int_), ("l2", np.int_), ("rand", np.float_)]
        )
        indices = np.argsort(lengths, order=("l1", "l2", "rand"))  # 排一下序
        # 每隔batch_size划分一下
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)

        return iter([i for batch in batches for i in batch])

    def __len__(self):

        return len(self.lengths)

