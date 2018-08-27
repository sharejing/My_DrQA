"""
    Time: 18-7-31 上午11:08
    Author: sharejing
    Description: 模型架构需要的参数

"""
import argparse

# Index of arguments concerning the core model architecture
MODEL_ARCHITECTURE = {
    'model_type', 'embedding_dim', 'hidden_size', 'doc_layers',
    'question_layers', 'rnn_type', 'concat_rnn_layers', 'question_merge',
    'use_qemb', 'use_in_question', 'use_pos', 'use_ner', 'use_lemma', 'use_tf'
}

# Index of arguments concerning the model optimizer/training
MODEL_OPTIMIZER = {
    'optimizer', 'learning_rate', 'momentum', 'weight_decay',
    'rnn_padding', 'dropout_rnn', 'dropout_rnn_output', 'dropout_emb',
    'max_len', 'grad_clipping'
}


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1", "y")


def add_model_args(parser):
    """
    往parser里添加模型架构的命令行参数
    :param parser:
    :return:
    """
    parser.register("type", "bool", str2bool)

    # 模型架构
    model = parser.add_argument_group("My DrQA model")
    model.add_argument("--embedding_dim", type=int, default=300,
                       help="Embedding size if embedding_file is not given")
    model.add_argument("--model_type", type=str, default="rnn",
                       help="Model architecture type")
    model.add_argument("--hidden_size", type=int, default=128,
                       help="Hidden size of RNN units")
    model.add_argument("--doc_layers", type=int, default=3,
                       help="Number of encoding layers for document")
    model.add_argument("--rnn_type", type=str, default="lstm",
                       help="RNN type: LSTM, GRU, or RNN")
    model.add_argument("--question_layers", type=int, default=3,
                       help="Number of encoding layers for question")

    # Model specific details
    detail = parser.add_argument_group("My DrQA model details")
    detail.add_argument("--use_in_question", type="bool", default=True,
                        help="Whether to use in_question_*features")
    detail.add_argument("--use_lemma", type="bool", default=True,
                        help="Whether to use lemma features")
    detail.add_argument("--use_pos", type="bool", default=True,
                        help="Whether to use pos features")
    detail.add_argument("--use_ner", type="bool", default=True,
                        help="Whether to use ner features")
    detail.add_argument("--use_tf", type="bool", default=True,
                        help="Whether to use term frequency features")
    detail.add_argument("--use_qemb", type="bool", default=True,
                        help="Whether to use weighted question embeddings")
    detail.add_argument("--concat_rnn_layers", type="bool", default=True,
                        help="Combine hidden states from each encoding layer")
    detail.add_argument("--question-merge", type=str, default="self_attn",
                        help="The way of computing the question representation")

    # Optimization details
    optim = parser.add_argument_group("My DrQA Optimization")
    optim.add_argument("--dropout_rnn", type=float, default=0.4,
                       help="Dropout rate for RNN states")
    optim.add_argument("--dropout_rnn_output", type="bool", default=True,
                       help="Whether to dropout the RNN output")
    optim.add_argument("--rnn_padding", type="bool", default=False,
                       help="Explicitly account for padding in RNN encoding")
    optim.add_argument("--dropout_emb", type=float, default=0.4,
                       help="Dropout rate for word embeddings")
    optim.add_argument("--optimizer", type=str, default="adamax",
                       help="Optimizer: sgd or adamax")
    optim.add_argument("--learning_rate", type=float, default=0.1,
                       help="Learning rate for SGD only")
    optim.add_argument("--momentum", type=float, default=0,
                       help="Momentum factor")
    optim.add_argument("--weight_decay", type=float, default=0,
                       help="Weight decay factor")
    optim.add_argument("--grad_clipping", type=float, default=10,
                       help="Gradient clipping")
    optim.add_argument("--max-len", type=int, default=15,
                       help="The max span allowed during decoding(预测答案)")


def get_model_args(args):
    """
    Filter args for model ones.
    From a args Namespace, return a new Namespace with *only* the args specific
    to the model architecture or optimization(i.e. the ones defined here.)
    :param args:
    :return:
    """
    global MODEL_ARCHITECTURE, MODEL_OPTIMIZER
    required_args = MODEL_ARCHITECTURE | MODEL_OPTIMIZER
    args_values = {k: v for k, v in vars(args).items() if k in required_args}

    return argparse.Namespace(**args_values)
