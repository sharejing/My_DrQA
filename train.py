"""
    Time: 18-7-31 上午10:42
    Author: sharejing
    Description: 预训练一个模型

"""
import argparse
from reader import config
import os
import torch
import logging
from reader import utils
from reader.model import DocReader
from reader import data
import torch.utils.data as tua
from reader import vector
import json
import numpy as np
from reader import validate

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1", "y")


def add_train_args(parser):
    """
    往parser里添加训练模型时的命令行参数
    :param parser:
    :return:
    """
    parser.register("type", "bool", str2bool)

    # 文件
    files = parser.add_argument_group("Filesystem")
    files.add_argument("--data_dir", type=str, default="data/dataset",
                       help="Directory of training/validation data")
    files.add_argument("--dev_json", type=str, default="SQuAD-v1.1-dev.json",
                       help="Unprocessed dev file to run validation while training on")
    files.add_argument("--train_file", type=str, default="SQuAD-v1.1-train-processed-corenlp.txt",
                       help="Preprocessed train file")
    files.add_argument("--dev_file", type=str, default="SQuAD-v1.1-dev-processed-corenlp.txt",
                       help="Preprocessed dev file")
    files.add_argument("--embedding_file", type=str, default="",
                       help="Space-separated pre-trained embeddings file")
    files.add_argument("--embedding_dir", type=str, default="data/embeddings",
                       help="Directory of pre-trained embedding file")
    files.add_argument("--model_dir", type=str, default="data/model",
                       help="Directory for saved models/checkpoints/logs")
    files.add_argument("--model_name", type=str, default="reader",
                       help="Unique model identifier")

    # 运行环境
    runtime = parser.add_argument_group("Environment")
    runtime.add_argument("--no_cuda", type="bool", default=True,
                         help="Train on CPU, even if GPUs are available")
    runtime.add_argument("--gpu", type=int, default=-1,
                         help="Run on a specific GPU")
    runtime.add_argument("--batch_size", type=int, default=32,
                         help="Batch size for training")
    runtime.add_argument("--data_workers", type=int, default=5,
                         help="Number of sub-processes for data loading")
    runtime.add_argument("--test_batch_size", type=int, default=128,
                         help="Batch size during validation/testing")
    runtime.add_argument("--num_epoches", type=int, default=32,
                         help="Train data iterations")
    runtime.add_argument("--random_seed", type=int, default=1013,
                         help="Random seed for all numpy/torch/cuda operations")
    runtime.add_argument("--parallel", type="bool", default=False,
                         help="Use DataParallel on all available GPUs")

    # 数据预处理
    preprocess = parser.add_argument_group("Pre-processing")
    preprocess.add_argument("--uncased_question", type="bool", default=False,
                            help="Question words will be lower-cased")
    preprocess.add_argument("--uncased_doc", type="bool", default=False,
                            help="Document words will be lower-cased")

    # General
    general = parser.add_argument_group("General")
    general.add_argument("--sort_by_len", type="bool", default=True,
                         help="Sort batches by length for speed")
    general.add_argument("--display-iter", type=int, default=25,
                         help="Log state after every <display-iter> epochs")
    general.add_argument("--official-eval", type="bool", default=True,
                         help="Validate with official SQuAD eval")
    general.add_argument("--valid_metric", type=str, default="f1",
                         help="The evaluation metric used for the final model selection")


def set_defaults(args):
    """
    需要初始化一些模型参数
    :param args:
    :return:
    """
    # 检查一些关键文件是否存在
    args.dev_json = os.path.join(args.data_dir, args.dev_json)
    if not os.path.isfile(args.dev_json):
        raise IOError("No such file: %s" % args.dev_json)

    args.train_file = os.path.join(args.data_dir, args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError("No such file: %s" % args.train_file)

    args.dev_file = os.path.join(args.data_dir, args.dev_file)
    if not os.path.isfile(args.dev_file):
        raise IOError("No such file: %s" % args.dev_file)

    if args.embedding_file:
        args.embedding_file = os.path.join(args.embedding_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError("No such file: %s" % args.embedding_file)

    # 设置日志文件和模型文件
    args.log_file = os.path.join(args.model_dir, args.model_name + ".txt")
    args.model_file = os.path.join(args.model_dir, args.model_name + ".mdl")

    # embedding
    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(" ")) - 1
        args.embedding_dim = dim

    return args


def init_from_scratch(args, train_exs, dev_exs):
    """
    New model, new data, new dictionary
    :param args:
    :param train_exs:
    :param dev_exs:
    :return:
    """
    # 根据数据集的标注建立特征词典
    logger.info("-" * 100)
    logger.info("Generate features")
    feature_dict = utils.build_feature_dict(args, train_exs)
    logger.info("Num features = %d" % len(feature_dict))
    logger.info(feature_dict)

    # 从训练集和开发集的documents和questions中建立词表
    logger.info("-" * 100)
    logger.info("Build dictionary")
    word_dict = utils.build_word_dict(args, train_exs + dev_exs)
    logger.info("Num words = %d" % len(word_dict))

    # 初始化模型
    model = DocReader(config.get_model_args(args), word_dict, feature_dict)

    return model


def train(args, data_loader, model, global_stats):
    """
    Run through one epoch of model training with the provided data loader.
    :param args:
    :param data_loader:
    :param model:
    :param global_stats:
    :return:
    """
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        train_loss.update(*model.update(ex))

        if idx % args.display_iter == 0:   # 在one epoch中每隔display_iter个batch_size就打印一次信息
            logger.info("train: epoch = %d | iter = %d/%d | " %
                        (global_stats["epoch"], idx, len(data_loader)) +
                        "loss = %.2f | elapsed time = %0.2f (s)" %
                        (train_loss.avg, global_stats["timer"].time()))
            train_loss.reset()

    logger.info("train: epoch %d done. Time for the epoch = %.2f (s)" %
                (global_stats["epoch"], epoch_time.time()))


def main(args):

    # ------------------------------------------------------------------------
    # Data
    # 加载数据
    logger.info("-" * 100)
    logger.info("Load data files")
    train_exs = utils.load_data(args, args.train_file, skip_no_answer=True)
    logger.info("Num train examples = %d" % len(train_exs))
    dev_exs = utils.load_data(args, args.dev_file)
    logger.info("Num dev examples = %d" % len(dev_exs))

    if args.official_eval:
        # id -> context and id -> [answers]
        # id -> context_offsets
        dev_texts, dev_answers = utils.load_text_and_answers(args.dev_json)
        dev_offsets = {ex["id"]: ex["offsets"] for ex in dev_exs}

    # -----------------------------------------------------------------------
    # Model
    logger.info("-" * 100)
    start_epoch = 0
    logger.info("Training model from scratch......")
    model = init_from_scratch(args, train_exs, dev_exs)

    # Set up optimizer
    model.init_optimizer()

    # Use the GPU?
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()

    # ----------------------------------------------------------------------
    # 划分数据，获得一个一个的batch_size
    # Two datasets: train and dev. If we sort by length it's faster
    logger.info("-" * 100)
    logger.info("Make data loaders")
    # train_dataset: document, features, question, start, end, ex["id"]
    train_dataset = data.ReaderDataset(train_exs, model, single_answer=True)  # 一行数据全变为索引
    if args.sort_by_len:
        train_sampler = data.SortedBatchSampler(train_dataset.lengths(),
                                                args.batch_size,
                                                shuffle=True)
    else:
        train_sampler = tua.Sampler.RandomSampler(train_dataset)
    train_loader = tua.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,  # 有序抽样
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
    )

    dev_dataset = data.ReaderDataset(dev_exs, model, single_answer=False)  # 一行数据全变为索引
    if args.sort_by_len:
        dev_sampler = data.SortedBatchSampler(dev_dataset.lengths(),
                                              args.test_batch_size,
                                              shuffle=True)
    else:
        dev_sampler = tua.Sampler.RandomSampler(dev_dataset)
    dev_loader = tua.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,  # 有序抽样
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
    )

    # -------------------------------------------------------------------------
    # 打印一下目前的参数
    logger.info("-" * 100)
    logger.info("Config:\n%s" %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # -------------------------------------------------------------------------
    # train/valid loop
    logger.info("-" * 100)
    logger.info("starting training......")
    stats = {"timer": utils.Timer(), "epoch": 0, "best_valid": 0}
    for epoch in range(start_epoch, args.num_epoches):
        stats["epoch"] = epoch

        # train
        train(args, train_loader, model, stats)

        # Validate unofficial (train)
        validate.validate_unofficial(args, train_loader, model, stats, mode="train")

        # Validate unofficial (dev)
        validate.validate_unofficial(args, dev_loader, model, stats, mode="dev")

        # Validate official
        if args.official_eval:
            result = validate.validate_official(args, dev_loader, model, stats,
                                       dev_offsets, dev_texts, dev_answers)

        # Save the model on best valid
        if result[args.valid_metric] > stats["best_valid"]:
            logger.info("Best valid: %s = %.2f (epoch %d, %d updates)" %
                        (args.valid_metric, result[args.valid_metric],
                         stats["epoch"], model.updates))
            model.save(args.model_file)
            stats["best_valid"] = result[args.valid_metric]


if __name__ == "__main__":

    # 需要用到的命令行参数
    parser = argparse.ArgumentParser(description="My DrQA")
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # 设置cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # 设置控制台日志打印
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s : [%(message)s]", "%m/%d/%Y %I:%M:%S %p")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    main(args)



