import copy
import csv
import logging
import math
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

from utils.mapper import ProblemMap


# def read_data_from_file(seq_file, pid2cid_file, cid2cname_file, length_filter=3):
#     """ Read interactions data from given csv file.
#
#
#     :param seq_file: Path of sequences file.
#                           File format(for each student):
#                           The first line is the number of problems a student attempted.
#                           The second line is the problem tag sequence.
#                           The third line is the response sequence.
#                           Example:
#                             15
#                             1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
#                             0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
#     :param pid2cid_file: Path of ProblemId-to-ConceptId file.
#                          File format:
#                          pid	cid
#
#     :param cid2cname_file: Path of ConceptId-to-ConceptName file.
#                            File format:
#                            cid	cname
#     :param length_filter: Ignore the sequence whose length is less than length_filter.
#     :return:
#         raw_data: List of samples.
#                   A sample is list with 3 elements:
#                     first elem(List[int]): problem tag sequence
#                     second elem(List[int]): concept tag sequence
#                     third elem(List[int]): response sequence
#         problem_map: Instance of ProblemMap
#     """
#
#     # set of cid and pid
#     cid_set = set()
#     pid_set = set()
#
#     # dict
#     pid2cid = {}
#     cid2cname = {}
#
#     # read the pid2cid file
#     for line in open(pid2cid_file, 'r'):
#         res = list(filter(None, line.strip().split("\t")))
#         if len(res) < 2:
#             try:
#                 pid2cid.setdefault(int(res[0]), -1)
#             except ValueError:
#                 pass
#         else:
#             pid2cid[int(res[0])] = int(res[1])
#             cid_set.add(int(res[1]))
#
#     # read the cid2cname file
#     for line in open(cid2cname_file, 'r'):
#         res = list(filter(None, line.strip().split("\t")))
#         if len(res) != 2:
#             try:
#                 cid2cname[int(res[1])] = '<unknown>'
#                 cid_set.add(int(res[1]))
#             except ValueError:
#                 pass
#         else:
#             cid2cname[int(res[1])] = res[0]
#             cid_set.add(int(res[1]))
#
#     # read the seq file
#     seq_file_rows = []
#     with open(seq_file, 'r') as f:
#         reader = csv.reader(f, delimiter=',')
#         for row in reader:
#             seq_file_rows.append(row)
#
#     raw_data = []
#     for i in range(0, len(seq_file_rows), 3):
#         # numbers of problem a student answered
#         seq_length = int(seq_file_rows[i][0])
#
#         # only keep student with at least length_filter records.
#         if seq_length < length_filter:
#             continue
#
#         problem_seq = seq_file_rows[i + 1]
#         correct_seq = seq_file_rows[i + 2]
#
#         try:
#             invalid_id_loc = problem_seq.index('')
#             del problem_seq[invalid_id_loc:]
#             del correct_seq[invalid_id_loc:]
#         except ValueError:
#             pass
#
#         # remove dirty data
#         if len(problem_seq) != len(correct_seq):
#             continue
#
#         # convert the sequence from string to int.
#         problem_seq = list(map(int, problem_seq))
#         correct_seq = list(map(int, correct_seq))
#         for u in correct_seq:
#             if u != 1 and u != 0:
#                 a = 10
#
#         pid_set |= set(problem_seq)
#
#         tup = [problem_seq, correct_seq]
#         raw_data.append(tup)
#
#     problem_map = ProblemMap.from_problem_id_set(pid_set)
#     problem_map.load_concept_id_set(cid_set)
#     problem_map.load_cid2cname(cid2cname)
#     problem_map.load_pid2cid(pid2cid)
#
#     for sample in raw_data:
#         sample[0] = problem_map.pid2pindices(sample[0])
#         sample.insert(1, [problem_map.pidx2cidx[pidx] for pidx in sample[0]])
#
#     return raw_data, problem_map

def read_data_from_file(data_path: str,
                        pid2pidx_path: str,
                        cid2cidx_path: str,
                        pidx2cidx_path: str,
                        cidx2cname_path: str,
                        length_filter: int = 3):
    """

    @param data_path: Path of data
                      Format(each sequence):
                          5
                          2 5 6 2 8
                          0 1 0 1 1
    @param pid2pidx_path: Path of pid2pidx.json
                          Format(each line):
                          2 1
    @param cid2cidx_path: Path of cid2cidx.json
                          Format(each line):
                          2 1
    @param pidx2cidx_path: Path of pidx2cidx.json
                           Format(each line):
                          5 8
    @param cidx2cname_path: Path of pidx2cidx.json
                            Format(each line):
                            5 Calculus
    @param length_filter: remove the sequences of length less than length_filter
    @return:
        raw_data (List[List[List[int]]]): List of all sequences.
                                     Each elem of raw_data is a sequence.
                                     A sequence is a list that contain 3 lists:
                                        problem_seq(List): List of problem idxs
                                        concept_seq(List): List of concept idxs
                                        answer_seq(List): List of answers (0/1)

        problem_map: Problem map of current dataset.
    """
    problem_map = ProblemMap()
    problem_map.load_pid2pidx(pid2pidx_path)
    problem_map.load_cid2cidx(cid2cidx_path)
    problem_map.load_pidx2cidx(pidx2cidx_path)
    problem_map.load_cidx2cname(cidx2cname_path)

    # read the seq file
    seq_file_rows = []
    with open(data_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            seq_file_rows.append(row)

    raw_data = []
    max_len = 0
    for i in range(0, len(seq_file_rows), 3):
        # numbers of problem a student answered
        seq_length = int(seq_file_rows[i][0])

        # only keep student with at least length_filter records.
        if seq_length < length_filter:
            continue

        problem_seq = seq_file_rows[i + 1]
        correct_seq = seq_file_rows[i + 2]
        max_len = max(max_len, len(problem_seq))

        try:
            invalid_id_loc = problem_seq.index('')
            del problem_seq[invalid_id_loc:]
            del correct_seq[invalid_id_loc:]
        except ValueError:
            pass

        # remove dirty data
        if len(problem_seq) != len(correct_seq):
            continue

        # convert the sequence from string to int.
        problem_seq = list(map(int, problem_seq))
        correct_seq = list(map(int, correct_seq))

        tup = [problem_seq, correct_seq]
        raw_data.append(tup)

    for sample in raw_data:
        sample[0] = problem_map.pid2pindices(sample[0])
        sample.insert(1, [problem_map.pidx2cidx[pidx] for pidx in sample[0]])

    logging.info("data:{} \t max_len:{}".format(data_path, max_len))
    return raw_data, problem_map


def pad_sequences(sequences: Tuple[List[List[int]], List[List[int]], List[List[int]]], pad_value: int = -1):
    """ Pad list of sequences according to the longest sequence in the batch.
        The paddings will be at the end of each sequence.

    @param sequences:
            sequences[0]: list of problem sequences, each element is a list of pidx
            sequences[1]: list of concept sequences, each element is a list of cidx
            sequences[2]: list of answer sequences, each element is a list of answers
    @param pad_value: padding value
    @return:
        sequences_padded (Tuple): (problem_seqs_padded, concept_seqs_padded, answer_seqs_padded)
        pad_value: padding value
    """

    problem_seqs, concept_seqs, answer_seqs = sequences
    max_len = max(len(s) for s in problem_seqs)
    problem_seqs_padded = []
    concept_seqs_padded = []
    answer_seqs_padded = []

    for seq in problem_seqs:
        problem_seqs_padded.append(seq + (max_len - len(seq)) * [pad_value])

    for seq in concept_seqs:
        concept_seqs_padded.append(seq + (max_len - len(seq)) * [pad_value])

    for seq in answer_seqs:
        answer_seqs_padded.append(seq + (max_len - len(seq)) * [pad_value])

    return (problem_seqs_padded, concept_seqs_padded, answer_seqs_padded), pad_value


def unpack_data(raw_data: List[List[List[int]]]):
    """Convert raw_data into list of pidx sequences, list of cidx and list of answer sequences
       (reverse sorted by length )

    @param raw_data: List of all sequences.
                     Each elem of raw_data is a sequence.
                     A sequence is a list that contain 3 lists:
                        problem_seq(List): List of problem idxs
                        concept_seq(List): List of concept idxs
                        answer_seq(List): List of answers (0/1)
    @return:
        problem_seqs: List of pidx sequences
        concept_seqs: List of cidx sequences
        answer_seqs: List of answer sequences
    """

    raw_data = sorted(raw_data, key=lambda e: len(e[0]), reverse=True)
    problem_seqs = [e[0] for e in raw_data]
    concept_seqs = [e[1] for e in raw_data]
    answer_seqs = [e[2] for e in raw_data]

    return problem_seqs, concept_seqs, answer_seqs


def batch_iter(raw_data, batch_size, shuffle=False):
    """ Yield batches of sequences reverse sorted by length (longest to shortest).

    @param raw_data: List of all sequences.
                     Each elem of raw_data is a sequence.
                     A sequence is a list that contain 3 lists:
                        problem_seq(List): List of problem idxs
                        concept_seq(List): List of concept idxs
                        answer_seq(List): List of answers (0/1)
    @param batch_size:
    @param shuffle: whether to randomly shuffle the dataset
    @yield:
        problem_seqs: List of pidx sequences
        concept_seqs: List of cidx sequences
        answer_seqs: List of answer sequences
    """
    batch_num = math.ceil(len(raw_data) / batch_size)
    index_array = list(range(len(raw_data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [raw_data[idx] for idx in indices]
        problem_seqs, concept_seqs, answer_seqs = unpack_data(examples)

        yield problem_seqs, concept_seqs, answer_seqs


def process_data(batch_data: Tuple[List[List[int]], List[List[int]], List[List[int]]],
                 num_problems: int,
                 num_concepts: int,
                 device: torch.device):
    """ Take a batch of data and convert it to torch.Tensor

    @param batch_data:
                batch_data[0](problem_seqs): List of pidx sequences
                batch_data[1](concept_seqs): List of cidx sequences
                batch_data[2](answer_seqs): List of answer sequences
    @param num_problems: Number of problems
    @param num_concepts: Number of concepts
    @param device: torch.device
    @return:
        response_data(torch.Tensor): [b, max_seq_len]
                       response_data[i][t] = problem_idx + {0, 1} * num_problems
        concept_data(torch.Tensor): [b, max_seq_len]
                       concept_data[i][t] = concept_idx
        seq_lengths(List[int]):sequence length of each sequence in the batch
    """

    problem_seqs, concept_seqs, answer_seqs = batch_data

    # Compute sequence lengths and max length
    seq_lengths = [len(s) for s in problem_seqs]

    # pad the data
    padded_data, pad_value = pad_sequences(batch_data)
    problem_seqs_padded, concept_seqs_padded, answer_seqs_padded = padded_data

    # compact the data and compute answered_problems_mask
    response_data = []
    concept_data = []
    for p_seq_padded, c_seq_padded, a_seq_padded in zip(problem_seqs_padded, concept_seqs_padded, answer_seqs_padded):
        response_seq = []
        concept_seq = []
        for pidx, cidx, answer in zip(p_seq_padded, c_seq_padded, a_seq_padded):
            if pidx == pad_value:
                response_seq.append(2 * num_problems)
                concept_seq.append(num_concepts)
            else:
                response_seq.append(pidx + answer * num_problems)
                concept_seq.append(cidx)
        response_data.append(response_seq)
        concept_data.append(concept_seq)

    response_data = torch.tensor(response_data, device=device)  # (batch_size, seq_len)
    concept_data = torch.tensor(concept_data, device=device)  # (batch_size, seq_len)

    return response_data, concept_data, seq_lengths


def response_data_to_problem_data(response_data: torch.Tensor, num_problems: int):
    """ Convert response_data to problem_data.

    @param response_data: [b, max_seq_len]
                          response_data[i][t] = problem_idx + {0, 1} * num_problems
    @param num_problems:  Number of problems
    @return:
           problem_data: [b, max_seq_len]
                         problem_data[i][t] = problem_idx
    """
    problem_data = copy.deepcopy(response_data)  # (batch_size, seq_len)
    problem_data = torch.where(problem_data >= num_problems, problem_data - num_problems, problem_data)
    return problem_data


def problem_data_to_cmask(problem_data: torch.Tensor, num_problems: int):
    """Convert problem_data to answered_problems_masks

    @param problem_data: [b, max_seq_len]
                          problem_data[i][t] = problem_idx
    @param num_problems: Number of problems
    @return:
        answered_problems_mask(torch.BoolTensor):
            [b, max_seq_len, num_problems]
            (answered_problems_mask[i][j][k] == True) => the ith student answered jth problem at kth step.
            (answered_problems_mask[i][j][k] == False) => !(the ith student answered jth problem at kth step)s
    """
    answered_problems_mask = F.one_hot(problem_data,
                                       num_classes=num_problems + 1)  # (batch_size, seq_len, num_problems+1)
    answered_problems_mask = answered_problems_mask[:, :, :-1].gt(0)  # (batch_size, seq_len, num_problems)
    return answered_problems_mask


def cmask2nmask(cur_mask: torch.BoolTensor):
    """Convert problem_data to answered_problems_masks to next_answered_problems_masks

    @param cur_mask:
               [b, max_seq_len, num_problems]
               (cur_mask[i][j][k] == True) => the ith student answered jth problem at kth step.
               (cur_mask[i][j][k] == False) => !(the ith student answered jth problem at kth step)s
    @return: next_mask:
             [b, max_seq_len, num_problems]
             (next_mask[i][j][k] == True) => the ith student answered jth problem at {k+1}th step.
             (next_mask[i][j][k] == False) => !(the ith student answered jth problem at {k+1}th step)s

    """
    batch_size, seq_len, num_problems = cur_mask.shape
    return torch.cat([cur_mask[:, 1:, :],
                      torch.zeros((batch_size, 1, num_problems),
                                  device=cur_mask.device).bool()]
                     , dim=1)


def response_data_to_one_hot(response_data: torch.Tensor, num_problems: int):
    """ Convert response_data to one-hot form.

    @param response_data: [b, max_seq_len]
                          response_data[i][t] = problem_idx + {0, 1} * num_problems
    @param num_problems: Number of problems
    @return:
        response_ont_hot: One-hot form of response_data.[batch_size, seq_len, 2*num_problems]

    """
    response_ont_hot = F.one_hot(response_data,
                                 num_classes=2 * num_problems + 1)  # (batch_size, seq_len, 2*num_problems+1)
    response_ont_hot = response_ont_hot[:, :, :-1]  # (batch_size, seq_len, 2*num_problems)
    return response_ont_hot


def concept_data_to_one_hot(concept_data: torch.Tensor, num_concepts: int):
    """ Convert concept_data to one-hot form.

    @param concept_data: [b, max_seq_len]
                          concept_data[i][t] = concept_idx
    @param num_concepts: Number of concepts
    @return:
        concept_ont_hot: One-hot form of concept_data.[batch_size, seq_len, num_concepts]

    """
    concept_ont_hot = F.one_hot(concept_data,
                                num_classes=num_concepts + 1)  # (batch_size, seq_len, num_concept+1)
    concept_ont_hot = concept_ont_hot[:, :, :-1]  # (batch_size, seq_len, num_concept)
    return concept_ont_hot


def mask2compact_mask(mask: torch.BoolTensor):
    """ Given a cmask or nmask and compact in problem dimension

    @param mask: [b, max_seq_len, num_problems]
    @return:
        compact_mask:
          [b, max_seq_len]ss
          compact_mask[i][j] = (mask[i][j][0] || mask[i][j][1] || ... || mask[i][j][num_problems-1])
          (compact_mask[i][j] = True) => ith student answered a problems at jth step(cmask)/ {j+1}th step(nmask)
    """
    num_slices = mask.shape[2]
    compact_mask = torch.zeros_like(mask[:, :, 0])
    for i in range(num_slices):
        compact_mask = compact_mask + mask[:, :, i].int()
    return compact_mask.bool()


if __name__ == "__main__":
    train_data_path = "../data/STATICS/STATICS_train.csv"
    test_data_path = "../data/STATICS/STATICS_test.csv"
    pid2pidx_path = "../data/STATICS/pid2pidx.json"
    cid2cidx_path = "../data/STATICS/cid2cidx.json"
    pidx2cidx_path = "../data/STATICS/pidx2cidx.json"
    cidx2cname_path = "../data/STATICS/cidx2cname.json"
    raw_data, problem_map = read_data_from_file(train_data_path, pid2pidx_path, cid2cidx_path, pidx2cidx_path,
                                                cidx2cname_path)
