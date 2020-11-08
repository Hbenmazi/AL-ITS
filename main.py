"""
Usage:
    main.py --model=<model_name> --dataset=<dataset_name>  [options]

Options:
    -h --help                               show this screen.
    --gpu=<int>                             use GPU [default: 0]
    --batch-size=<int>                      batch size [default: 4]
    --train-ratio=<float>                   ratio of training set [default: 0.8]
    --hidden-size=<int>                     hidden size [default: 100]
    --dropout=<float>                       dropout [default: 0.5]
    --lambda-r=<float>                      lambda_r [default: 0.01]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --max-epoch=<int>                       max epoch [default: 200]
    --max-patience=<int>                    wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --lr=<float>                            learning rate [default: 0.01]
    --model-save-to=<file>                  model save path [default: outputs/model/]

"""

import logging

from docopt import docopt
import itertools
import math
import os
import torch
from torch import optim, nn
from KTModels.models import DKT, DKTEnhanced, PredictionConsistentBCELoss
from utils.data_utils import read_data_from_file, batch_iter, process_data, unpack_data
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, accuracy_score
import copy
import numpy as np
import tqdm


def evaluate(eval_data, num_problems, num_concepts, model):
    problem_seqs, concept_seqs, answer_seqs = eval_data

    # process the input data
    response_data, concept_data, seq_lengths = process_data(eval_data, num_problems, num_concepts, device=model.device)

    # get y_next_true + y_cur_true
    next_answer_seqs = copy.deepcopy(answer_seqs)
    for n_ans_seq in next_answer_seqs:
        del n_ans_seq[0]

    y_next_true = list(itertools.chain.from_iterable(next_answer_seqs))
    y_next_true = torch.tensor(y_next_true, dtype=torch.float, device=model.device)
    y_cur_true = list(itertools.chain.from_iterable(answer_seqs))
    y_cur_true = torch.tensor(y_cur_true, dtype=torch.float, device=model.device)

    # forward +  y_next_pred + y_cur_pred
    y_next_pred, y_cur_pred, batch_future_pred, batch_concept_mastery = model(response_data, concept_data, seq_lengths)

    y_next_pred_label = np.where(y_next_pred.cpu().numpy() <= 0.5, 0, 1)
    auc_n = roc_auc_score(y_next_true.cpu(), y_next_pred.cpu())
    auc_c = roc_auc_score(y_cur_true.cpu(), y_cur_pred.cpu())
    acc = accuracy_score(y_next_true.cpu(), y_next_pred_label)
    rmse = np.sqrt(mean_squared_error(y_next_true.cpu(), y_next_pred.cpu()))
    mae = mean_absolute_error(y_next_true.cpu(), y_next_pred.cpu())
    return auc_n, auc_c, acc, rmse, mae


def test(args: dict):
    model_name = args['--model']
    model_path = args['model-path']
    data_path = args['test-data-path']
    pid2pidx_path = args['pid2pidx-path']
    cid2cidx_path = args['cid2cidx-path']
    pidx2cidx_path = args['pidx2cidx-path']
    cidx2cname_path = args['cidx2cname-path']

    if model_name == 'DKT':
        model = DKT.load(model_path)
    elif model_name == 'DKTEnhanced':
        model = DKTEnhanced.load(model_path)
    else:
        raise ValueError

    if args['--gpu'] is not None:
        device = torch.device("cuda:" + args['--gpu'] if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    logging.info("Using ".format(device))
    model.to(device)

    test_dataset, problem_map = read_data_from_file(data_path,
                                                    pid2pidx_path,
                                                    cid2cidx_path,
                                                    pidx2cidx_path,
                                                    cidx2cname_path)
    unpacked_test_dataset = unpack_data(test_dataset)
    num_problems = len(problem_map)
    num_concepts = problem_map.num_concepts

    print('=' * 10 + 'TEST' + '=' * 10)
    print("Model:", model.__name__())
    print("Data:", data_path)
    model.eval()
    with torch.no_grad():
        auc_n, auc_c, acc, rmse, mae = evaluate(unpacked_test_dataset, num_problems, num_concepts, model)
        print("test_auc(n) %.6f" % auc_n)
        print("test_auc(c) %.6f" % auc_c)
        print("test_acc    %.6f" % acc)
        print("test_rmse   %.6f" % rmse)
        print("test_mae    %.6f" % mae)


def train(args: dict):
    model_name = args['--model']
    batch_size = int(args['--batch-size'])
    max_patience = int(args['--max-patience'])
    max_num_trial = int(args['--max-num-trial'])
    lr_decay = float(args['--lr-decay'])
    train_ratio = float(args['--train-ratio'])
    model_save_to = args['--model-save-to']
    max_epoch = int(args['--max-epoch'])

    model_path = args['model-path']
    train_data_path = args['train-data-path']
    pid2pidx_path = args['pid2pidx-path']
    cid2cidx_path = args['cid2cidx-path']
    pidx2cidx_path = args['pidx2cidx-path']
    cidx2cname_path = args['cidx2cname-path']

    raw_data, problem_map = read_data_from_file(train_data_path, pid2pidx_path, cid2cidx_path, pidx2cidx_path,
                                                cidx2cname_path)

    num_problems = len(problem_map)
    num_concepts = problem_map.num_concepts

    train_size = int(train_ratio * len(raw_data))
    val_size = len(raw_data) - train_size

    train_dataset, val_dataset = random_split(raw_data, [train_size, val_size])
    unpacked_train_dataset = unpack_data(train_dataset)
    unpacked_val_dataset = unpack_data(val_dataset)

    if model_name == 'DKT':
        model = DKT(num_problems=num_problems, hidden_size=int(args['--hidden-size']),
                    dropout_rate=float(args['--dropout']))
        criterion = torch.nn.BCELoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=float(args['--lr']))
    elif model_name == "DKTEnhanced":
        model = DKTEnhanced(problem_map=problem_map, hidden_size=int(args['--hidden-size']),
                            dropout_rate=float(args['--dropout']))
        criterion = PredictionConsistentBCELoss(lambda_r=float(args['--lambda-r']))
        optimizer = optim.Adam(model.parameters(), lr=float(args['--lr']))
    else:
        raise ValueError("wrong value of model_name[{}]".format(model_name))

    if args['--gpu'] is not None:
        device = torch.device("cuda:" + args['--gpu'] if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    logging.info("Using {}".format(device))
    model.to(device)

    # for p in model.parameters():
    #     p.data.uniform_(-0.1, 0.1)

    train_iter = 0
    patience = 0
    num_trial = 0
    hist_valid_scores = []

    # print statistic infomation
    with torch.no_grad():
        model.eval()
        t_auc_n, t_auc_c, t_acc, t_rmse, t_mae = evaluate(unpacked_train_dataset, num_problems, num_concepts, model)
        v_auc_n, v_auc_c, v_acc, v_rmse, v_mae = evaluate(unpacked_val_dataset, num_problems, num_concepts, model)
        print("Init:")
        print("valid_auc(n) %.6f\ttrain_auc(n) %.6f\t" % (v_auc_n, t_auc_n))
        print("valid_auc(c) %.6f\ttrain_auc(c) %.6f\t" % (v_auc_c, t_auc_c))
        print("valid_acc    %.6f\ttrain_acc    %.6f\t" % (v_acc, t_acc))
        print("valid_rmse   %.6f\ttrain_rmse   %.6f\t" % (v_rmse, t_rmse))
        print("valid_mae    %.6f\ttrain_mae    %.6f\t" % (v_mae, t_mae))

    batch_num = math.ceil(len(train_dataset) / batch_size)
    for epoch in range(max_epoch):  # loop over the dataset multiple times
        for batch_data in tqdm.tqdm(batch_iter(train_dataset, batch_size=batch_size),
                                    desc="Epoch[{}]".format(epoch + 1), total=batch_num):
            train_iter += 1

            model.train()
            problem_seqs, concept_seqs, answer_seqs = batch_data

            # get y_next_true ans y_cur_true
            next_answer_seqs = copy.deepcopy(answer_seqs)
            for n_ans_seq in next_answer_seqs:
                del n_ans_seq[0]

            y_next_true = list(itertools.chain.from_iterable(next_answer_seqs))
            y_next_true = torch.tensor(y_next_true, dtype=torch.float, device=model.device)
            y_cur_true = list(itertools.chain.from_iterable(answer_seqs))
            y_cur_true = torch.tensor(y_cur_true, dtype=torch.float, device=model.device)

            # process the input data
            response_data, concept_data, seq_lengths = process_data(batch_data, num_problems, num_concepts,
                                                                    device=model.device)

            # forward + backward + optimize
            y_next_pred, y_cur_pred, batch_future_pred, batch_concept_mastery = model(response_data, concept_data,
                                                                                      seq_lengths)

            if model_name == 'DKT':
                loss = criterion(y_next_pred, y_next_true)
            elif model_name == 'DKTEnhanced':
                loss = criterion(y_next_pred, y_next_true, y_cur_pred, y_cur_true)
            else:
                raise ValueError

            loss.backward()

            # clip gradient
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args['--clip-grad']))
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

        # print statistic information
        with torch.no_grad():
            model.eval()
            t_auc_n, t_auc_c, t_acc, t_rmse, t_mae = evaluate(unpacked_train_dataset, num_problems, num_concepts, model)
            v_auc_n, v_auc_c, v_acc, v_rmse, v_mae = evaluate(unpacked_val_dataset, num_problems, num_concepts, model)
            print("valid_auc(n) %.6f\ttrain_auc(n) %.6f\t" % (v_auc_n, t_auc_n))
            print("valid_auc(c) %.6f\ttrain_auc(c) %.6f\t" % (v_auc_c, t_auc_c))
            print("valid_acc    %.6f\ttrain_acc    %.6f\t" % (v_acc, t_acc))
            print("valid_rmse   %.6f\ttrain_rmse   %.6f\t" % (v_rmse, t_rmse))
            print("valid_mae    %.6f\ttrain_mae    %.6f\t" % (v_mae, t_mae))

        # check for early stop
        is_better = len(hist_valid_scores) == 0 or v_auc_n > max(hist_valid_scores)
        hist_valid_scores.append(v_auc_n)

        if is_better:
            patience = 0
            if not os.path.exists(model_save_to):
                os.makedirs(model_save_to)
            model.save(model_path)

            # also save the optimizers' state
            torch.save(optimizer.state_dict(), model_path + '.optim')

        elif patience < int(max_patience):
            patience += 1
            print('hit patience %d' % patience)

            if patience == int(max_patience):
                num_trial += 1
                print('hit #%d trial' % num_trial)
                if num_trial == int(max_num_trial):
                    print('early stop!')
                    break

                # decay lr, and restore from previously best checkpoint
                lr = optimizer.param_groups[0]['lr'] * float(lr_decay)

                # load model
                params = torch.load(model_path, map_location=lambda storage, loc: storage)
                model.load_state_dict(params['state_dict'])
                model = model.to(device)

                optimizer.load_state_dict(torch.load(model_path + '.optim'))

                # set new lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # reset patience
                patience = 0

    print('Finished Training')


def main():
    """ Main func.
    """
    args = docopt(__doc__)

    legal_model = ['DKT', 'DKTEnhanced']
    legal_datasets = ['STATICS', 'assist2009', 'assist2015']
    assert args['--model'] in legal_model, "Model should be in {}".format(legal_model)
    assert args['--dataset'] in legal_datasets, "Dataset should be in {}".format(legal_datasets)

    if args['--dataset'] == 'STATICS':
        args['train-data-path'] = "data/STATICS/STATICS_train.csv"
        args['test-data-path'] = "data/STATICS/STATICS_test.csv"
        args['pid2pidx-path'] = "data/STATICS/pid2pidx.json"
        args['cid2cidx-path'] = "data/STATICS/cid2cidx.json"
        args['pidx2cidx-path'] = "data/STATICS/pidx2cidx.json"
        args['cidx2cname-path'] = "data/STATICS/cidx2cname.json"
    elif args['--dataset'] == 'assist2009':
        args['train-data-path'] = "data/assist2009_updated/assist2009_updated_train.csv"
        args['test-data-path'] = "data/assist2009_updated/assist2009_updated_test.csv"
        args['pid2pidx-path'] = "data/assist2009_updated/pid2pidx.json"
        args['cid2cidx-path'] = "data/assist2009_updated/cid2cidx.json"
        args['pidx2cidx-path'] = "data/assist2009_updated/pidx2cidx.json"
        args['cidx2cname-path'] = "data/assist2009_updated/cidx2cname.json"
    elif args['--dataset'] == 'assist2015':
        args['train-data-path'] = "data/assist2015/assist2015_train.csv"
        args['test-data-path'] = "data/assist2015/assist2015_test.csv"
        args['pid2pidx-path'] = "data/assist2015/pid2pidx.json"
        args['cid2cidx-path'] = "data/assist2015/cid2cidx.json"
        args['pidx2cidx-path'] = "data/assist2015/pidx2cidx.json"
        args['cidx2cname-path'] = "data/assist2015/cidx2cname.json"

    args['model-path'] = args['--model-save-to'] + args['--model'] + '-' + args['--dataset'] + '.bin'
    train(args)
    test(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    # train(model_name,
    #       train_data_path,
    #       pid2pidx_path,
    #       cid2cidx_path,
    #       pidx2cidx_path,
    #       cidx2cname_path,
    #       train_ratio,
    #       train_batch_size,
    #       max_patience,
    #       max_num_trial,
    #       lr_decay,
    #       model_save_to)
    #
    # test(model_name,
    #      model_save_to + model_name + '-' + train_data_path.split(sep='/')[1] + '.bin',
    #      test_data_path,
    #      pid2pidx_path,
    #      cid2cidx_path,
    #      pidx2cidx_path,
    #      cidx2cname_path)
