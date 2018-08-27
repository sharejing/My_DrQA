"""
    Time: 18-8-4 下午4:58
    Author: sharejing
    Description:

"""
import logging
from reader import utils
import torch
import re
import string
from collections import Counter

logger = logging.getLogger(__name__)


# Validation loops. Include both "unofficial" and "official" functions that
# use different metrics and implementations
def validate_unofficial(args, data_loader, model, global_stats, mode):
    """
    Run one full unofficial validation
    Unofficial = doesn't use SQuAD script
    :param args:
    :param data_loader:
    :param model:
    :param global_stats:
    :param mode:
    :return:
    """
    eval_time = utils.Timer()
    start_acc = utils.AverageMeter()
    end_acc = utils.AverageMeter()
    exact_match = utils.AverageMeter()

    # Make predictions
    examples = 0
    for ex in data_loader:
        batch_size = ex[0].size(0)
        pred_start, pred_end, _ = model.predict(ex)
        target_start, target_end = ex[-3:-1]

        # We get metrics for independent start/end and joint start/end
        accuracies = eval_accuracies(pred_start, target_start, pred_end, target_end)
        start_acc.update(accuracies[0], batch_size)
        end_acc.update(accuracies[1], batch_size)
        exact_match.update(accuracies[2], batch_size)

        # for getting train accuracies, sample ex max 10k
        examples += batch_size
        if mode == "train" and examples >= 1e4:
            break
    logger.info("%s valid unofficial: epoch = %d | start_is_match = %.2f | " %
                (mode, global_stats["epoch"], start_acc.avg) +
                "end_is_match = %.2f | start_and_end_is_exact_match = %.2f | use_examples = %d | " %
                (end_acc.avg, exact_match.avg, examples) +
                "valid_use_time = %.2f (s)" % eval_time.time())

    return {"start_and_end_is_exact_match": exact_match.avg}


def eval_accuracies(pred_start, target_start, pred_end, target_end):
    """
    An unofficial evaluation helper.
    Compute exact start/end complete match accuracies for a batch.
    :param pred_start: batch_size
    :param target_start: batch_size
    :param pred_end: batch_size
    :param target_end: batch_size
    :return:
    """
    # Covert 1D tensors to lists of lists (compatibility)
    if torch.is_tensor(target_start):
        target_start = [[e] for e in target_start]
        target_end = [[e] for e in target_end]

    # Compute accuracies from targets
    batch_size = len(pred_start)
    start = utils.AverageMeter()
    end = utils.AverageMeter()
    em = utils.AverageMeter()
    for i in range(batch_size):
        # Start matched
        if pred_start[i] in target_start[i]:
            start.update(1)
        else:
            start.update(0)

        # End matched
        if pred_end[i] in target_end[i]:
            end.update(1)
        else:
            end.update(0)

        # Both start and end match
        if any([1 for _s, _e in zip(target_start[i], target_end[i])
                if _s == pred_start[i] and _e == pred_end[i]]):
            em.update(1)
        else:
            em.update(0)

    return start.avg * 100, end.avg * 100, em.avg * 100


# 使用f1评价
def validate_official(args, data_loader, model, global_stats, offsets, texts, answers):
    """
    Run one full offical validation. Uses exact spans and same exact match/F1 score
    computation as in the SQuAD script.
    :param args:
    :param data_loader:
    :param model:
    :param global_stats:
    :param offsets: dev: id -> context_offsets
    :param texts:   dev: id -> context
    :param answers: dev: id -> [answers]
    :return:
    """
    eval_time = utils.Timer()
    f1 = utils.AverageMeter()
    exact_match = utils.AverageMeter()

    # Run through examples
    examples = 0
    for ex in data_loader:
        ex_id, batch_size = ex[-1], ex[0].size(0)
        pred_start, pred_end, _ = model.predict(ex)

        for i in range(batch_size):
            start_offset = offsets[ex_id[i]][pred_start[i][0]][0]
            end_offset = offsets[ex_id[i]][pred_end[i][0]][1]
            prediction = texts[ex_id[i]][start_offset:end_offset]

            # Compute metrics
            ground_truths = answers[ex_id[i]]

            # EM(每一个question的预测答案与标准答案的完全匹配分数)
            em_value = metric_max_over_over_ground_truths(exact_match_score,
                                                          prediction, ground_truths)
            exact_match.update(em_value)

            # f1(每一个question的预测答案与标准答案的f1分数)
            f1_value = metric_max_over_over_ground_truths(f1_score,
                                                          prediction, ground_truths)
            f1.update(f1_value)

        examples += batch_size

    logger.info("dev valid official: epoch = %d | EM = %.2f | " %
                (global_stats["epoch"], exact_match.avg * 100) +
                "F1 = %.2f | use_examples = %d | valid_use_time = %.2f (s)" %
                (f1.avg * 100, examples, eval_time.time()))

    return {"exact_match": exact_match.avg * 100, "f1": f1.avg * 100}


# 1
def metric_max_over_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    Given a prediction_answer and multiple valid answers, return the score of
    the best prediction_answer - answer_n pair given a metric function
    :param metric_fn:
    :param prediction:
    :param ground_truths:
    :return:
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)

    return max(scores_for_ground_truths)


# 2
def exact_match_score(prediction, ground_truth):
    """
    Check if the prediction is a (soft) exact match with the ground truth
    :param prediction:
    :param ground_truth:
    :return:
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# 3
def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace
    :param s:
    :return:
    """
    # 移除冠词
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    # 移除多余的空白
    def white_space_fix(text):
        return ' '.join(text.split())

    # 移除英文标点符号
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    # 全部小写
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# 4
def f1_score(prediction, groud_truth):
    """
    Compute the geometric mean of precision ans recall for answer tokens
    :param prediction:
    :param groud_truth:
    :return:
    """
    prediction_tokens = normalize_answer(prediction).split()
    groud_truth_tokens = normalize_answer(groud_truth).split()
    common = Counter(prediction_tokens) & Counter(groud_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    # 正确率
    precision = 1.0 * num_same / len(prediction_tokens)
    # 召回率
    recall = 1.0 * num_same / len(groud_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


