from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from utils.data_utils import response_data_to_one_hot, response_data_to_problem_data, problem_data_to_cmask, \
    cmask2nmask
from utils.mapper import ProblemMap


class InitialStateLearnableLSTM(nn.Module):
    r""" Init state and cell learnable LSTM.

       Args:
           input_size: The number of expected features in the input `x`
           hidden_size: The number of features in the hidden state `h`
           num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
               would mean stacking two LSTMs together to form a `stacked LSTM`,
               with the second LSTM taking in outputs of the first LSTM and
               computing the final results. Default: 1
           bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
               Default: ``True``
           batch_first: If ``True``, then the input and output tensors are provided
               as (batch, seq, feature). Default: ``False``
           dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
               LSTM layer except the last layer, with dropout probability equal to
               :attr:`dropout`. Default: 0
           bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
       """

    def __init__(self, *args, **kwargs):
        super(InitialStateLearnableLSTM, self).__init__()
        self.lstm = nn.LSTM(*args, **kwargs)

        self.num_layers = self.lstm.num_layers
        self.hidden_size = self.lstm.hidden_size
        self.num_directions = 2 if self.lstm.bidirectional else 1
        self.h0 = nn.Parameter(data=torch.zeros(self.num_layers * self.num_directions, 1, self.hidden_size),
                               requires_grad=True)
        self.c0 = nn.Parameter(data=torch.zeros(self.num_layers * self.num_directions, 1, self.hidden_size),
                               requires_grad=True)
        self.reset_parameters()

    def forward(self, rnn_input, prev_states=None):
        batch_size = rnn_input.shape[0] if self.lstm.batch_first else rnn_input.shape[1]

        if prev_states is None:
            state_size = (self.num_layers * self.num_directions, batch_size, self.hidden_size)
            init_h = self.h0.expand(*state_size).contiguous()
            init_c = self.c0.expand(*state_size).contiguous()
            prev_states = (init_h, init_c)

        return self.lstm(rnn_input, prev_states)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.c0, gain=nn.init.calculate_gain('relu'))


class DKT(nn.Module):
    """ Deep Knowledge Tracing Model

        -Original Paper：
            Piech, Chris et al. “Deep Knowledge Tracing.” Advances in Neural Information Processing Systems.
            Vol. 2015-January. Neural information processing systems foundation, 2015. 505–513. Print.
    """

    def __init__(self, num_problems: int, hidden_size: int, dropout_rate: float):
        """

        @param num_problems: Number of problems
        @param hidden_size: Size of hidden state of RNN
        @param dropout_rate:
        """
        super(DKT, self).__init__()
        self.num_problems = num_problems
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.rnn = nn.LSTM(input_size=2 * self.num_problems, hidden_size=self.hidden_size, batch_first=True)
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=hidden_size, out_features=self.num_problems)
        self.reset_parameters()

    @staticmethod
    def __name__():
        return 'DKT'

    def forward(self, response_data: torch.Tensor, concept_data: torch.Tensor, seq_lengths: List[int]):
        """ Take a batch of data to train the model,
            return predictions of each time step(only for answered problem of current and next time step),
            predictions of future time step.

        @param response_data: [b, max_seq_len]
                              response_data[i][t] = problem_idx + {0, 1} * num_problems
        @param concept_data: [b, max_seq_len]
                             concept_data[i][t] = concept_idx
        @param seq_lengths: sequence length of each sequence in the batch
        @return:
            y_next_pred: [sum(seq_lengths)]
                         predictions of each time step for current problems

            y_cur_pred: [sum(seq_lengths) - batch_size]
                        predictions of each time step for next problems

            batch_future_pred: [batch_size, num_problems]
                               predictions of all problems of all students in next future time step

            batch_concept_mastery: None

        """
        # concert to one-hot and forward
        batch_size = response_data.shape[0]
        one_hot_data = response_data_to_one_hot(response_data, num_problems=self.num_problems)
        hiddens, h_n = self.rnn(pack_padded_sequence(one_hot_data.float(), seq_lengths, batch_first=True))
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True)  # hiddens(b, seq_len, hidden_size)
        dropout_hiddens = self.dropout(hiddens)
        outputs = torch.sigmoid(self.fc(dropout_hiddens))  # outputs(b, seq_len, num_problems)

        # extract data
        problem_data = response_data_to_problem_data(response_data, num_problems=self.num_problems)
        cmask = problem_data_to_cmask(problem_data, num_problems=self.num_problems)
        nmask = cmask2nmask(cmask)
        y_next_pred = torch.masked_select(outputs, nmask)  # (sum(seq_lengths))
        y_cur_pred = torch.masked_select(outputs, cmask)  # (sum(seq_lengths) - batch_size)

        batch_future_pred = []
        for i in range(batch_size):
            batch_future_pred.append(outputs[i, seq_lengths[i] - 1, :])
        batch_future_pred = torch.stack(batch_future_pred, dim=0)  # (batch_size, num_problems)
        batch_concept_mastery = None

        return y_next_pred, y_cur_pred, batch_future_pred, batch_concept_mastery

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return next(self.parameters()).device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path: path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = DKT(**args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path: path to the model
        """

        params = {
            'args': dict(num_problems=self.num_problems, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() < 2:
                nn.init.uniform_(p.data, -0.1, 0.1)
            else:
                nn.init.xavier_uniform_(p.data)


class DKTEnhanced(nn.Module):
    def __init__(self, problem_map: ProblemMap,
                 hidden_size: int = 200,
                 problem_code_dim: int = 100,
                 response_code_dim: int = 100,
                 concept_code_dim: int = 25,
                 overall_dim: int = 50,
                 dropout_rate: float = 0.4):
        """

        @param problem_map: ProblemMap of current dataset
        @param hidden_size: Size of hidden state of RNN
        @param problem_code_dim:  Size of problem embedding
        @param response_code_dim:  Size of response embedding
        @param concept_code_dim:  Size of concept embedding
        @param overall_dim: Size of overall vector
        @param dropout_rate:
        """

        super(DKTEnhanced, self).__init__()

        self.problem_map = problem_map
        self.num_problems = len(problem_map)
        self.num_concepts = problem_map.num_concepts
        self.hidden_size = hidden_size
        self.problem_code_dim = problem_code_dim
        self.response_code_dim = response_code_dim
        self.concept_code_dim = concept_code_dim
        self.overall_dim = overall_dim
        self.dropout_rate = dropout_rate

        self.response_embedding = nn.Embedding(num_embeddings=2 * self.num_problems + 1,
                                               embedding_dim=self.response_code_dim,
                                               padding_idx=2 * self.num_problems)
        self.problem_embedding = nn.Embedding(num_embeddings=self.num_problems + 1,
                                              embedding_dim=self.problem_code_dim,
                                              padding_idx=self.num_problems)
        self.concept_embedding = nn.Embedding(num_embeddings=self.num_concepts + 1,
                                              embedding_dim=self.concept_code_dim,
                                              padding_idx=self.num_concepts)
        self.concept_memory = nn.Parameter(data=torch.zeros(self.concept_code_dim, self.num_concepts),
                                           requires_grad=True)
        # self.rnn = nn.LSTM(input_size=self.problem_code_dim + self.response_code_dim, hidden_size=self.hidden_size)
        self.rnn = InitialStateLearnableLSTM(input_size=self.problem_code_dim + self.response_code_dim,
                                             hidden_size=self.hidden_size)

        self.dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.fc1 = nn.Linear(in_features=hidden_size + problem_code_dim, out_features=self.overall_dim)
        self.fc2 = nn.Linear(in_features=self.overall_dim, out_features=1)

        self.reset_parameters()

    def forward(self, response_data, concept_data, seq_lengths):
        """ Take a batch of data to train the model,
            return predictions of each time step(only for answered problem of current and next time step),
            predictions of future time step.

        @param response_data: [b, max_seq_len]
                              response_data[i][t] = problem_idx + {0, 1} * num_problems
        @param concept_data: [b, max_seq_len]
                             concept_data[i][t] = concept_idx
        @param seq_lengths: sequence length of each sequence in the batch
        @return:
            y_next_pred: [sum(seq_lengths)]
                         predictions of each time step for current problems

            y_cur_pred: [sum(seq_lengths) - batch_size]
                        predictions of each time step for next problems

            batch_future_pred: [batch_size, num_problems]
                               predictions of all problems of all students in next future time step

            batch_concept_mastery: [batch_size, num_concepts]
                                   concept mastery of all students
        """
        batch_size, seq_len = response_data.shape
        problem_data = response_data_to_problem_data(response_data, num_problems=self.num_problems)

        batch_cur_prediction = []
        batch_next_prediction = []
        batch_future_pred = []
        batch_concept_mastery = []
        for i in range(batch_size):
            # forward
            cur_pembed = self.problem_embedding(problem_data[i, :][:seq_lengths[i]])  # (seq_len, pro_code_dim)
            cur_rembed = self.response_embedding(response_data[i, :][:seq_lengths[i]])  # (seq_len, res_code_dim)
            cur_prembed = torch.cat([cur_pembed, cur_rembed], dim=1)  # (seq_len, pro_code_dim+res_code_dim)
            cur_cembed = self.concept_embedding(concept_data[i, :][:seq_lengths[i]])  # (seq_len, concept_code_dim)
            cur_beta = torch.matmul(cur_cembed, self.concept_memory)  # (seq_len, num_concepts)
            cur_beta = F.softmax(cur_beta, dim=1)

            rnn_input = torch.bmm(cur_beta.unsqueeze(dim=2),
                                  cur_prembed.unsqueeze(dim=1))  # (seq_len, num_concepts, pro_code_dim+res_code_dim)

            hiddens, _ = self.rnn(rnn_input)  # (seq_len, num_concepts, hidden_size)

            # predict for cur
            with torch.no_grad():
                cur_s = torch.cat([cur_pembed, torch.bmm(cur_beta.unsqueeze(dim=1), hiddens).squeeze(dim=1)],
                                  dim=1)  # (seq_len, pro_code_dim+hidden_size)
                cur_prediction = torch.sigmoid(self.fc2(F.relu(self.fc1(cur_s)))).squeeze(dim=1)  # (seq_len)
                batch_cur_prediction.append(cur_prediction)

            # predict for next
            next_pembed = self.problem_embedding(problem_data[i, :][1:seq_lengths[i]])  # (seq_len-1, pro_code_dim)
            next_cembed = self.concept_embedding(concept_data[i, :][1:seq_lengths[i]])  # (seq_len-1, embed_dim)
            next_beta = torch.matmul(next_cembed, self.concept_memory)  # (seq_len-1, num_concepts)
            next_beta = F.softmax(next_beta, dim=1)

            next_s = torch.cat([next_pembed, torch.bmm(next_beta.unsqueeze(dim=1), hiddens[:-1, :, :]).squeeze(dim=1)],
                               dim=1)  # (seq_len-1, pro_code_dim+hidden_size)
            next_prediction = torch.sigmoid(self.fc2(F.relu(self.fc1(next_s)))).squeeze(dim=1)  # (seq_len-1)
            batch_next_prediction.append(next_prediction)

            with torch.no_grad():
                last_hidden = hiddens[-1, :, :]  # (num_concepts, hidden_size)

                # future_pred
                all_pembed = self.problem_embedding(
                    torch.tensor(list(self.problem_map.pidx2cidx.keys()),
                                 device=self.device))  # (num_problems, pro_code_dim)
                all_cembed = self.concept_embedding(
                    torch.tensor(list(self.problem_map.pidx2cidx.values()),
                                 device=self.device))  # (num_problems, embed_dim)
                all_beta = torch.matmul(all_cembed, self.concept_memory)  # (num_problems, num_concepts)
                all_beta = F.softmax(all_beta, dim=1)  # (num_problems, num_concepts)

                all_s = torch.cat(
                    [all_pembed, torch.bmm(all_beta.unsqueeze(dim=1),
                                           last_hidden.unsqueeze(dim=0).expand(self.num_problems, -1, -1)).squeeze(
                        dim=1)],
                    dim=1)  # (num_problems, pro_code_dim+hidden_size)
                future_pred = torch.sigmoid(self.fc2(F.relu(self.fc1(all_s)))).squeeze(dim=1)  # (num_problems)
                batch_future_pred.append(future_pred)

                # Concept mastery

                concept_s = torch.cat(
                    [torch.zeros((self.num_concepts, self.problem_code_dim), device=self.device), last_hidden],
                    dim=1)  # (num_concepts, pro_code_dim+hidden_size)
                concept_mastery = torch.sigmoid(self.fc2(F.relu(self.fc1(concept_s)))).squeeze(dim=1)  # (num_concepts)
                batch_concept_mastery.append(concept_mastery)

        y_cur_pred = torch.cat(batch_cur_prediction, dim=0)
        y_next_pred = torch.cat(batch_next_prediction, dim=0)
        batch_future_pred = torch.stack(batch_future_pred, dim=0)
        batch_concept_mastery = torch.stack(batch_concept_mastery, dim=0)

        return y_next_pred, y_cur_pred, batch_future_pred, batch_concept_mastery

    @staticmethod
    def __name__():
        return 'DKTEnhanced'

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return next(self.parameters()).device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path: path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = DKTEnhanced(**args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path: path to the model
        """
        params = {
            'args': dict(
                problem_map=self.problem_map,
                hidden_size=self.hidden_size,
                problem_code_dim=self.problem_code_dim,
                response_code_dim=self.response_code_dim,
                concept_code_dim=self.concept_code_dim,
                overall_dim=self.overall_dim,
                dropout_rate=self.dropout_rate),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() < 2:
                nn.init.uniform_(p.data, -0.1, 0.1)
            else:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))


class PredictionConsistentBCELoss(nn.Module):
    """ PredictionConsistentBCELoss

    """

    def __init__(self, lambda_r):
        super().__init__()
        self.lambda_r = lambda_r

    def forward(self, y_next_pred, y_next_true, y_cur_pred, y_cur_true):
        assert (y_next_pred.ge(0) & y_next_pred.le(1)).all()
        assert (y_cur_pred.ge(0) & y_cur_pred.le(1)).all()

        L = F.binary_cross_entropy(y_next_pred, y_next_true, reduction='mean')
        r = F.binary_cross_entropy(y_cur_pred, y_cur_true, reduction='mean')

        return L + self.lambda_r * r
