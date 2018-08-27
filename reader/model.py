"""
    Time: 18-8-1 下午3:44
    Author: sharejing
    Description: DrQA Document Reader Model
                加载整个网络，训练，预测全是这个类

"""
from .rnn_reader import RnnDocReader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import copy
import logging

logger = logging.getLogger(__name__)


class DocReader(object):
    """
    High level model that handles initializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """
    # 初始化
    def __init__(self, args, word_dict, feature_dict, state_dict=None, normalize=True):

        self.args = args
        self.word_dict = word_dict
        self.args.vocab_size = len(word_dict)
        self.feature_dict = feature_dict
        self.args.num_features = len(feature_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        # Building network. If normalize is false, scores are not normalized
        # 0-1 per paragraph (no softmax) 这个if似乎没有
        if args.model_type == "rnn":
            self.network = RnnDocReader(args, normalize)   # DrQA的网络
        else:
            raise RuntimeError("Unsupported model: %s" % args.model_type)

    def init_optimizer(self, state_dict=None):
        """
        Initialize an optimizer for the free parameters of the network
        :param state_dict:
        :return:
        """
        parameters = [p for p in self.network.parameters()]
        if self.args.optimizer == "sgd":
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "adamax":
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError("Unsupported optimizer: %s" % self.args.optimizer)

    # Learning
    def update(self, ex):
        """
        Forward a batch of examples; step the optimizer to update weights
        :param ex:代表一个batch_size,总共含8个元素：
            1) document word indices: batch_size * len_doc
            2) document word features indices: batch_size * len_doc * nfeat
            3) document padding mask: batch_size * len_doc
            4) question word indices: batch_size * len_question
            5) question padding mask: batch_size * len_question
            6) answer_start_label: batch_size
            7) answer_end_label: batch_size
            8) ids: (list)batch_size
        :return:
        """
        if not self.optimizer:
            raise RuntimeError("No optimizer set")

        # Train mode(训练模式)
        self.network.train()

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True)) for e in ex[:5]]
            target_start = Variable(ex[5].cuda(async=True))
            target_end = Variable(ex[6].cuda(async=True))

        else:
            inputs = [e if e is None else Variable(e) for e in ex[:5]]
            target_start = Variable(ex[5])
            target_end = Variable(ex[6])

        # Run forward
        score_start, score_end = self.network(*inputs)  # batch_size * len_doc

        # Compute loss and accuracies
        loss = F.nll_loss(score_start, target_start) + F.nll_loss(score_end, target_end)

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm(self.network.parameters(), self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        return loss.data[0], ex[0].size(0)

    # Prediction
    def predict(self, ex, candidates=None, top_n=1, async_pool=None):
        """
        Forward a batch of examples only to get predictions
        :param ex: a batch_size
        :param candidates:
        :param top_n:
        :param async_pool:
        :return:
        """
        # Eval mode(评价模式)
        self.network.eval()

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True), volatile=True) for e in ex[:5]]
        else:
            inputs = [e if e is None else Variable(e, volatile=True) for e in ex[:5]]

        # Run forward
        score_start, score_end = self.network(*inputs)

        # Decode predictions
        score_start = score_start.data.cpu()
        score_end = score_end.data.cpu()
        if candidates:
            args = (score_start, score_end, candidates, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode_candidates, args)
            else:
                return self.decode_candidates(*args)
        else:
            args = (score_start, score_end, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode, args)
            else:
                return self.decode(*args)

    @staticmethod
    def decode(score_start, score_end, top_n=1, max_len=None):
        """
        Take argmax of constrained score_start * score_end.
        :param score_start: independent start predictions: batch_size * len_doc
        :param score_end: independent end predictions: batch_size * len_doc
        :param top_n: number of top scored pairs to take
        :param max_len: max span length to consider
        :return:
        """
        pred_start = []
        pred_end = []
        pred_score = []
        max_len = max_len or score_start.size(1)
        for i in range(score_start.size(0)):
            # Outer product of scores to get full p_s * p_e matrix
            scores = torch.ger(score_start[i], score_end[i])  # len_doc * len_doc

            # Zero out negative length and over-length span scores
            scores.triu_().tril(max_len - 1)

            # Take argmax or top n
            scores = scores.numpy()
            scores_flat = scores.flatten()
            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            start_idx, end_idx = np.unravel_index(idx_sort, scores.shape)
            pred_start.append(start_idx)
            pred_end.append(end_idx)
            pred_score.append(scores_flat[idx_sort])
        # pred_start: (list)batch_size | pred_end: (list)batch_size | pred_score: (list)batch_size
        return pred_start, pred_end, pred_score

    @staticmethod
    def decode_candidates(score_start, score_end, candidates, top_n=1, max_len=None):
        """
        Take argmax of constrained score_start * score_end. Except only consider spans
        that are in the candidates list.
        :param score_start:
        :param score_end:
        :param candidates:
        :param top_n:
        :param max_len:
        :return:
        """

    # Saving and loading
    def save(self, filename):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        params = {
            "state_dict": state_dict,
            "word_dict": self.word_dict,
            "feature_dict": self.feature_dict,
            "args": self.args
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logging.warning("WARN: Saving model failed.... continuing anyway.")

    # Runtime
    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """
        Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        :return:
        """
        self.parallel = True
        self.network = nn.DataParallel(self.network)









