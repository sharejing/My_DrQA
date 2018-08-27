"""
    Time: 18-8-1 下午4:24
    Author: sharejing
    Description: Definitions of model layers/NN modules
                 DrQA由哪些模块组成

"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class StackedBRNN(nn.Module):
    """
    Stacked Bi-directional RNN.
    Differs from standard Pytorch library in that it has the option to save and
    concat the hidden states between layers.(i.e. the output hidden size for each
    sequence input is num_layers * hidden_size)
    """
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()  # 相当于一个list

        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """
        Encode either padded or non-padded sequences.
        Can choose to either handle or ignore variable length sequences.
        Always handle in eval.
        :param x:
            document: batch_size * len_doc * (word embedding + aligned question embedding + manual features)
        :param x_mask:
            document: batch_size * len_doc
        :return:
            document: batch_size * len_doc * (hidden_size * 2 * doc_layers)
        """
        if x_mask.data.sum() == 0:
            # No padding necessary
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            # Pad if we care or if its during eval
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """
        Faster encoding that ignores any padding
        :param x:
        :param x_mask:
        :return:
        """
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)  # len_doc * batch_size * (word embedding + aligned question embedding + manual features)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        """
        For document's outputs:
                    [len_doc * batch_size * (word embedding + aligned question embedding + manual features),
                     len_doc * batch_size * (hidden_size * 2),
                     len_doc * batch_size * (hidden_size * 2),
                     len_doc * batch_size * (hidden_size * 2)]
        """

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)  # len_doc * batch_size * (hidden_size * 2 * num_layers)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)  # batch_size * len_doc * (hidden_size * 2 * num_layers)

        return output

    def _forward_padded(self, x, x_mask):
        """
        Slower (significantly), but more precise, encoding that handles padding.
        :param x:
        :param x_mask:
        :return:
        """
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input, rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers ro take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class SeqAttnMatch(nn.Module):
    """
    Aligned question embedding
    Given sequences X and Y, match sequence Y to each element in X
    X -> document
    Y -> question
    """
    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """

        :param x: batch_size * x_len * embedding_dim
        :param y: batch_size * y_len * embedding_dim
        :param y_mask: batch_size * y_len
        :return:
                matched_seq: batch_size * x_len * embedding_dim
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)  # batch_size * x_len * embedding_dim
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)  # batch_size * y_len * embedding_dim
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))  # batch_size * x_len * y_len

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())  # batch_size * x_len * y_len
        scores.data.masked_fill_(y_mask.data, -float("inf"))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=-1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))   # batch_size * x_len * y_len

        # Take weighted average
        matched_seq = alpha.bmm(y)  # batch_size * x_len * embedding_dim

        return matched_seq


class LinearSeqAttn(nn.Module):
    """
    Self attention over a sequence
    """
    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """

        :param x:
            question: batch_size * question_len * (hidden_size * 2 * question_layers)
        :param x_mask:
            question: batch_size * question_len
        :return:
            question: batch_size * question_len
        """
        x_flat = x.view(-1, x.size(-1))  # (len_question * 2) * (hidden_size * 2 * question_layers)
        scores = self.linear(x_flat).view(x.size(0), x.size(1))  # batch_size * question_len
        scores.data.masked_fill_(x_mask.data, -float("inf"))
        alpha = F.softmax(scores, dim=-1)

        return alpha


class BilinearSeqAttn(nn.Module):
    """
    A bi-linear attention layer over a sequence X w.r.t y
    """
    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize

        # if identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """

        :param x: 特指document: batch_size * len_doc * (hidden_size * 2 * doc_layers)
        :param y: 特值question: batch_size * (hidden_size * 2 * question_layers)
        :param x_mask: 特指document: batch_size * len_doc
        :return:
            特指document: batch_size * len_doc
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float("inf"))
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                alpha = F.log_softmax(xWy, dim=-1)
            else:
                # Otherwise 0-1 probabilities
                alpha = F.softmax(xWy, dim=-1)
        else:
            alpha = xWy.exp()

        return alpha


# Functional
def uniform_weights(x, x_mask):
    """
    Return uniform weights over non-masked x(a sequence of vectors)
    :param x: batch_size * len_question * (hidden_size * 2 * question_layers)
    :param x_mask: batch_size * len_question
    :return: batch_size * (hidden_size * 2 * question_layers)
    """
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())

    return alpha


def weighted_avg(x, weights):
    """
    Return a weighted average of x(a sequence of vectors)
    :param x: batch_size * len_question * (hidden_size * 2 * question_layers)
    :param weights: batch_size * len_question
    :return: batch_size * (hidden_size * 2 * question_layers)
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)





