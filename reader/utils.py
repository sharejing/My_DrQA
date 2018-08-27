"""
    Time: 18-7-31 下午9:08
    Author: sharejing
    Description: 一些处理函数

"""
import json
from .data import Dictionary
import time


# 数据加载函数
def load_data(args, filename, skip_no_answer=False):
    """
    Load examples form preprocessed file. one example per line, JSON encoded.
    :param args:
    :param filename:
    :param skip_no_answer:
    :return:
    """
    # Load JSON lines
    with open(filename) as f:
        examples = [json.loads(line) for line in f]

    # Make case insensitive?
    if args.uncased_question or args.uncased_doc:
        for ex in examples:
            if args.uncased_question:
                ex["question"] = [w.lower() for w in ex["question"]]
            if args.uncased_doc:
                ex["document"] = [w.lower() for w in ex["document"]]

    # Skip unparsed(start/end) examples
    if skip_no_answer:
        examples = [ex for ex in examples if len(ex["answers"]) > 0]

    return examples


# 建立字典
def build_feature_dict(args, examples):
    """
    Index features (one hot) from fields in examples and options
    :param args:
    :param examples:
    :return:
    """
    feature_dict = {}

    def _insert(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

    # Exact match features(有3个)
    # We use three simple binary features, indicating whether pi can be exactly matched
    # to one question word in q, either in its original, lowercase or lemma form
    if args.use_in_question:
        _insert("in_question")
        _insert("in_question_uncased")
        if args.use_lemma:
            _insert("in_question_lemma")

    # Token features
    # (1) Part of speech tag features
    if args.use_pos:
        for ex in examples:
            for w in ex["pos"]:
                _insert("pos=%s" % w)

    # (2) Named entity tag features
    if args.use_ner:
        for ex in examples:
            for w in ex["ner"]:
                _insert("ner=%s" % w)

    # (3) Term frequency features
    if args.use_tf:
        _insert("tf")

    return feature_dict


def build_word_dict(args, examples):
    """
    Return a dict from question and document words in provided examples
    :param args:
    :param examples:
    :return:
    """
    word_dict = Dictionary()
    for w in load_words(args, examples):
        word_dict.add(w)

    return word_dict


def load_words(args, examples):
    """
    Iterate and index all the words in examples(documents + questions)
    :param args:
    :param examples:
    :return:
    """
    words = set()

    def _insert(iterable):
        for w in iterable:
            w = Dictionary.normalize(w)
            words.add(w)

    for ex in examples:
        _insert(ex["document"])
        _insert(ex["question"])

    return words


def load_text_and_answers(filename):
    """
    Load the paragraphs only of a SQuAD dataset. Store as qid -> text. qid -> [answers]
    :param filename:
    :return:
    """
    # Load Json file
    with open(filename) as f:
        examples = json.load(f)["data"]

    texts = {}
    ans = {}
    for article in examples:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                texts[qa["id"]] = paragraph["context"]
                ans[qa["id"]] = list(map(lambda x: x["text"], qa["answers"]))

    return texts, ans


# Utility classes
class Timer(object):
    """
    Computes elapsed time
    """
    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()

        return self

    def resume(self):
        if not self.running:
            self.running = False
            self.total += time.time() - self.start

        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start

        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start

        return self.total


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count