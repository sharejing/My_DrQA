"""
    Time: 18-8-1 下午4:06
    Author: sharejing
    Description: Implementation of the RNN based DrQA reader
                 DrQA整个网络架构

"""
import torch.nn as nn
from . import layers
import torch.nn.functional as F
import torch


# 网络
class RnnDocReader(nn.Module):
    RNN_TYPES = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}

    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__()

        # Store config
        self.args = args

        # Word embeddings
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=0)

        # Projection for attention weighted question(Aligned question embedding)
        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to RNN: word embedding + Aligned question embedding + manual features
        doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb:
            doc_input_size += args.embedding_dim

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.question_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers      # 768
            question_hidden_size *= args.question_layers  # 768

        # Question merging(与transformer里面的self attention是否有区别)
        if args.question_merge not in ["avg", "self_attn"]:
            raise NotImplementedError("merge_mode = %s" % args.question_merge)
        if args.question_merge == "self_attn":
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Bi-linear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize
        )

    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        """

        :param x1:
            document word indices: batch_size * len_doc
        :param x1_f:
            document word features indices: batch_size * len_doc * nfeat
        :param x1_mask:
            document padding mask: batch_size * len_doc
        :param x2:
            question word indices: batch_size * len_question
        :param x2_mask:
            question padding mask: batch_size * len_question
        :return:
            batch_size * len_doc
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)  # batch_size * len_doc * word_embedding
        x2_emb = self.embedding(x2)  # batch_size * len_question * word_embedding

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = F.dropout(x1_emb, p=self.args.dropout_emb,
                               training=self.training)
            x2_emb = F.dropout(x2_emb, p=self.args.dropout_emb,
                               training=self.training)

        # Form document encoding inputs
        drnn_input = [x1_emb]

        # Add Aligned question embedding
        if self.args.use_qemb:
            x2_aligned_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input.append(x2_aligned_emb)

        # Add manual features(Exact match有3个, POS, NER, TF)
        if self.args.num_features > 0:
            drnn_input.append(x1_f)

        # Encode document with RNN
        # batch_size * len_doc * (hidden_size * 2 * doc_layers)
        doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask)

        """
        以上document的编码结束：
            doc_hiddens: batch_size * len_doc * (hidden_size * 2 * doc_layers)
            1) 输入： word embedding + Aligned question embedding + manual features
            2) 经过3层双向的LSTM，拼接每一层的hidden size得到document的表示： doc_hiddens
        思考可以改进的地方：
            1) manual features还是不变，但可以增加一些其他更多的特征
                对于英文(中文)： POS NER TF.....
            2) document也可以经过一个self attention
            
        """

        # Encode question with RNN + merge hiddens
        """
        question_hiddens: 
            batch_size * len_question * (hidden_size * 2 * question_layers)
        q_merge_weights(avg):
            batch_size * (hidden_size * 2 * question_layers)
        q_merge_weights(self_attn):
            batch_size * len_question
        question_hidden:
            batch_size * (hidden_size * 2 * question_layers)
        """
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        if self.args.question_merge == "avg":
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.args.question_merge == "self_attn":
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        """
        以上question编码结束：
            question_hidden:batch_size * (hidden_size * 2 * question_layers)
            1) 输入： word embedding
            2) 经过3层双向的LSTM，拼接每一层的hidden size得到question的表示： question_hiddens
            3) 再经过一个self attention层得到最终的question_hidden
        思考可以改进的地方：
            1) question也可以考虑特征的拼接
            2) 这里的self attention与transformer的self attention有什么区别？
        """

        # Predict start and end positions
        # start_scores/end_scores:
        #   batch_size * len_doc
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)

        """
            一些其他改进点：
                1) 特征不采用拼接而采用 1.求和方式 2.多模态融合方式
                2) document再经过一层self attention
                3) document和question经过3层LSTM后：
                    让document的表示与question的表示经过一次多模态融合
                    document: batch_size * len_doc * (hidden_size * 2 * doc_layers)
                    question: batch_size * len_question * (hidden_size * 2 * question_layers)
                    result = multi_modal_layer(document, question)
                    result的size依然是：batch_size * len_doc * (hidden_size * 2 * doc_layers)
                    
                
        """

        return start_scores, end_scores






