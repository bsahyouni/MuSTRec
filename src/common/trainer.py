import os
import itertools
from symbol import parameters

import torch
import torch.optim as optim
from torch.optim import Adam
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from time import time
from logging import getLogger
import wandb

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator
from utils.seq_metrics import recall_at_k, ndcg_k, mean_average_precision_at_k, precision_at_k


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model, model2=None, mg=False, user_embeddings=None, item_embeddings=None):
        super(Trainer, self).__init__(config, model)

        if model2 is not None:
            self.model2 = model2
        else:
            self.model2 = None
        if user_embeddings is not None and item_embeddings is not None:
            self.user_embeddings = user_embeddings
            self.item_embeddings = item_embeddings
        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']

        self.start_epoch = 0
        self.cur_step = 0
        self.max_len = 50

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.best_valid_seq = None
        self.best_test_seq = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        # fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = config['learning_rate_scheduler']  # check zero?
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None
        self.mg = mg
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']
        self.beta = config['beta']
        self.multimodal_switchoff = 0

        # New initializations from trainers.py
        self.cuda_condition = torch.cuda.is_available() and not config['no_cuda']
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        if self.cuda_condition:
            self.model.cuda()

        # # Initialize Adam optimizer
        # betas = (config['adam_beta1'], config['adam_beta2'])
        # breakpoint()
        # self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, betas=betas,
        #                       weight_decay=self.weight_decay)

        self.logger.info(f"Total Parameters: {sum([p.nelement() for p in self.model.parameters()])}")
        self.omega = 1

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        # Get parameters from models
        model_params = list(self.model.parameters())
        model2_params = list(self.model2.parameters()) if self.model2 is not None else []

        # Get parameters from embeddings
        embedding_params = list(self.user_embeddings.parameters()) + list(self.item_embeddings.parameters())

        # Check if embeddings are in model parameters
        embeddings_in_model = all(any(p is ep for p in model_params) for ep in embedding_params)
        # For model2, only check item_embeddings
        item_embedding_params = list(self.item_embeddings.parameters())
        embeddings_in_model2 = all(
            any(p is ep for p in model2_params) for ep in item_embedding_params) if self.model2 else True

        print("Embeddings in model parameters:", embeddings_in_model)
        print("Embeddings in model2 parameters:", embeddings_in_model2)

        parameters = model_params + model2_params + embedding_params

        #remove duplicate parameters
        parameters = list({id(p): p for p in parameters}.values())

        # if self.model2 is not None:
        #     parameters = list(self.model.parameters()) + list(self.model2.parameters()) + \
        #              list(self.user_embeddings.parameters()) + list(self.item_embeddings.parameters())
        # else:
        #     parameters = list(self.model.parameters()) + \
        #                              list(self.user_embeddings.parameters()) + list(self.item_embeddings.parameters())

        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(parameters, lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        if not self.req_training:
            return 0.0, []
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        seq_loss = None
        loss_batches = []
        # first_loss_batches = []
        # second_loss_batches = []
        str_code = "train"
        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(enumerate(train_data),
                                  desc="Mode_%s:%d" % (str_code, epoch_idx),
                                  total=len(train_data),
                                  bar_format="{l_bar}{r_bar}")
        for batch_idx, interaction in rec_data_iter:
            self.optimizer.zero_grad()
            second_inter = torch.cat((interaction[:2], interaction[53:]), dim=0).clone()
            if self.multimodal_switchoff == 1:
                losses = 0
            else:
                losses = loss_func(torch.cat((interaction[:2], interaction[53:]), dim=0))
            if self.model2 is not None:
                third_inter = interaction.clone()
                # if epoch_idx < 60:
                #     seq_loss = 0
                # else:
                seq_loss = self.SequentialLossEval(third_inter, batch_idx)
                wandb.log({'multimodal_loss': losses,
                           'sequential_loss': seq_loss})

            if seq_loss is not None:
                losses = losses + self.omega*seq_loss
                wandb.log({'combined_loss': losses})

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))

            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if self._check_nan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0), torch.tensor(0.0)

            if self.mg and batch_idx % self.beta == 0:
                first_loss = self.alpha1 * loss
                wandb.log({'combined_loss_scaled': first_loss})
                #first_loss_batches.append(first_loss)
                first_loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                losses = loss_func(second_inter)
                if isinstance(losses, tuple):
                    loss = sum(losses)
                else:
                    loss = losses

                if self._check_nan(loss):
                    self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                    return loss, torch.tensor(0.0)
                second_loss = -1 * self.alpha2 * loss
                wandb.log({'second_loss': second_loss})
                second_loss.backward()
            else:
                loss.backward()

            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())
            # for test
            # if batch_idx == 0:
            #    break
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data, phase):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result, sequential_scores = self.evaluate(valid_data, phase)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result, sequential_scores

    def _check_nan(self, loss):
        if torch.isnan(loss):
            # raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """

        wandb.init(project="MuSTRec", name=f'{self.omega}_baby')
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            self.model.pre_epoch_processing()
            # train_loss, first_loss_batches, second_loss_batches = self._train_epoch(train_data, epoch_idx)
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            # if self.model2 is not None:
            #     sequential_loss_batch = self.SequentialLossEval(epoch_idx, train_data, train=True)
            # for loss1, loss2, loss3 in zip(first_loss_batches, second_loss_batches, sequential_loss_batch):
            #     # Apply backward for the first loss (this happens for every batch)
            #     combined_loss = loss1 + self.omega * loss3
            #     combined_loss.backward()
            #     if self.clip_grad_norm:
            #         clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            #     self.optimizer.step()
            #     self.optimizer.zero_grad()

                # # Apply backward for the second loss only when it's not None (self.mg and batch_idx % self.beta == 0)
                # if loss2 is not None:
                #     loss2.backward()
                #     self.optimizer.step()
                #     self.optimizer.zero_grad()

            if torch.is_tensor(train_loss):
                # get nan loss
                break
            # for param_group in self.optimizer.param_groups:
            #    print('======lr: ', param_group['lr'])
            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # # Saving embeddings every 100 epochs
            # if (epoch_idx + 1) % 100 == 0:
            #     # Using the forward method to obtain embeddings
            #     with torch.no_grad():
            #         self.model.eval()  # Set model to evaluation mode to disable dropout, etc.
            #         adjacency_matrix = self.model.masked_adj
            #         u_g_embeddings, i_g_embeddings = self.model.forward(adjacency_matrix)
            #         torch.save(u_g_embeddings, f'user_embeddings_epoch_{epoch_idx + 1}.pt')
            #         torch.save(i_g_embeddings, f'item_embeddings_epoch_{epoch_idx + 1}.pt')
            #         self.model.train()  # Set model back to training mode

            # eval: To ensure the test result is the best model under validation data, set self.eval_step == 1
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                self.config['train_matrix'] = self.config['valid_rating_matrix']
                valid_score, valid_result, valid_sequential = self._valid_epoch(valid_data, phase='valid')
                # self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                #     valid_score, self.best_valid_score, self.cur_step,
                #     max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_sequential[12], self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                # test
                self.config['train_matrix'] = self.config['test_rating_matrix']
                _, test_result, test_sequential = self._valid_epoch(test_data, phase='test')

                valid_metrics = {'valid/' + k: v for k, v in valid_result.items()}
                test_metrics = {'test/' + k: v for k, v in test_result.items()}
                metrics = {**valid_metrics, **test_metrics, 'epoch': epoch_idx}
                wandb.log(metrics)

                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))
                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_valid_seq = valid_sequential
                    self.best_test_upon_valid = test_result
                    self.best_test_seq = test_sequential

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        wandb.finish()
        return self.best_valid_seq, self.best_valid_result, self.best_test_seq, self.best_test_upon_valid
        # return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid

    @torch.no_grad()
    def evaluate(self, eval_data, phase, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()
        # self.config['train_matrix'] = self.config['valid_rating_matrix'] if is_test else self.config['test_rating_matrix']

        # batch full users
        batch_matrix_list = []
        pred_list = []
        answer_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            # predict: interaction without item ids
            batched_data_free = batched_data[:2]
            scores = self.model.full_sort_predict(batched_data_free)
            masked_items = batched_data_free[1]
            # mask out pos items
            scores[masked_items[0], masked_items[1]] = -1e10
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
            batch_pred_list, batch_answers = self.SequentialLossEval(batched_data, batch_idx, train=False)
            pred_list.extend(batch_pred_list)
            answer_list.extend(batch_answers)

            # Convert lists to arrays
        pred_list = np.array(pred_list)
        answer_list = np.array(answer_list)

        # After all batches, compute metrics over the entire epoch
        sequential_scores, _ = self.get_full_sort_score(answer_list, pred_list, phase)

        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx), sequential_scores

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)

    def SequentialLossEval(self, bat, batch_idx, train=True):

        if train:
            self.model2.train()
            batch_losses = []
            # rec_loss = 0.0

            # str_code = "train" if train else "test"
            # rec_data_iter = tqdm.tqdm(enumerate(bat),
            #                           desc="Mode_%s:%d" % (str_code, epoch),
            #                           total=len(bat),
            #                           bar_format="{l_bar}{r_bar}")

            #for i, batch in rec_data_iter:
            # Determine the batch size and sequence length
            batch_size = bat.shape[1]
            seq_length = self.max_len  # Replace with your actual sequence length

            # Initialize index for slicing
            idx = 0

            # Extract user_tensor
            user_tensor = bat[idx]  # Shape: (batch_size,)
            idx += 1

            # Extract item_tensor (if needed)
            item_tensor = bat[idx]  # Shape: (batch_size,)
            idx += 1

            # Extract seq_tensor and transpose back to (batch_size, seq_length)
            seq_tensor = bat[idx:idx + seq_length].transpose(0, 1)  # Shape: (batch_size, seq_length)
            idx += seq_length

            # Extract answer_tensor
            answer_tensor = bat[idx]  # Shape: (batch_size,)
            idx += 1

            # Extract neg_answer_tensor
            neg_answer_tensor = bat[idx]  # Shape: (batch_size,)
            idx += 1

            #batch = tuple(t.to(self.device) for t in batch)

            user_ids, input_ids, answers, neg_answer, same_target = user_tensor, seq_tensor, answer_tensor, neg_answer_tensor, torch.empty(256, 0, device='cuda:0', dtype=torch.int64)

            # result = self.check_data_coherence('/home/bs00826/Downloads/MMRec-master/data/sports/new_sports.txt', user_ids, input_ids, answers)
            # print("Data coherence check result:", result)

            loss = self.model2.calculate_loss(input_ids, answers, neg_answer, same_target, user_ids)
            #batch_losses.append(loss)
            # rec_loss += loss.item()
            # post_fix = {
            #     "epoch": epoch,
            #     "rec_loss": '{:.4f}'.format(rec_loss / len(rec_data_iter)),
            # }
            #
            # if (epoch + 1) % 1 == 0:
            #     self.logger.info(str(post_fix))
            return loss

        else:
            self.model2.eval()
            pred_list = []
            answer_list = []

            # str_code = "train" if train else "test"
            # rec_data_iter = tqdm.tqdm(enumerate(bat),
            #                           desc="Mode_%s:%d" % (str_code, epoch),
            #                           total=len(bat),
            #                           bar_format="{l_bar}{r_bar}")

            #for i, batch in rec_data_iter:
            # Determine the batch size and sequence length
            #batch_size = bat.shape[1]
            seq_length = self.max_len  # Replace with your actual sequence length

            # Initialize index for slicing
            idx = 0

            user_ids, input_ids, answers = bat[0], bat[2], bat[3]
            recommend_output = self.model2.predict(input_ids, user_ids)
            recommend_output = recommend_output[:, -1, :]  # 推荐的结果

            rating_pred = self.predict_full(recommend_output)
            rating_pred = rating_pred.cpu().data.numpy().copy()
            batch_user_index = user_ids.cpu().numpy()

            try:
                rating_pred[self.config['train_matrix'][batch_user_index].toarray() > 0] = 0
            except:  # bert4rec
                rating_pred = rating_pred[:, :-1]
                rating_pred[self.config['train_matrix'][batch_user_index].toarray() > 0] = 0

            # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
            # argpartition time complexity O(n)  argsort O(nlogn)
            # The minus sign "-" indicates a larger value.
            ind = np.argpartition(rating_pred, -20)[:, -20:]
            # Take the corresponding values from the corresponding dimension
            # according to the returned subscript to get the sub-table of each row of topk
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
            # Sort the sub-tables in order of magnitude.
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
            # retrieve the original subscript from index again
            batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

            # Return per-batch predictions and answers without computing metrics
            return batch_pred_list.tolist(), answers.cpu().data.numpy().tolist()

            # if batch_idx == 0:
            #     pred_list.extend(batch_pred_list)
            #     answer_list.extend(answers.cpu().data.numpy())
            # else:
            #     pred_list.extend(batch_pred_list)
            #     answer_list.extend(answers.cpu().data.numpy())
            #
            # pred_list = np.array(pred_list)
            # answer_list = np.array(answer_list)
            #
            # return self.get_full_sort_score(answer_list, pred_list)

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model2.item_embeddings.weight
        # [batch hidden_size ]
        # import pdb; pdb.set_trace()
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def get_full_sort_score(self, answers, pred_list, phase):
        recall, ndcg, precision, map = [], [], [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
            precision.append(precision_at_k(answers, pred_list, k))
            map.append(mean_average_precision_at_k(answers, pred_list, k))

        post_fix = {
            "HR@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HR@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HR@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        # Combine the metrics into a single list in the order of 'metrics'
        metrics_values = []
        for idx in range(len([5, 10, 15, 20])):
            metrics_values.extend([recall[idx], ndcg[idx], precision[idx], map[idx]])

        # Log metrics with phase prefix
        metric_names = ['HR@5', 'NDCG@5', 'Precision@5', 'MAP@5',
                        'HR@10', 'NDCG@10', 'Precision@10', 'MAP@10',
                        'HR@15', 'NDCG@15', 'Precision@15', 'MAP@15',
                        'HR@20', 'NDCG@20', 'Precision@20', 'MAP@20']
        for idx, metric_name in enumerate(metric_names):
            if wandb.run is not None:
                wandb.log({f"{phase}_Sequential/{metric_name}": metrics_values[idx]})
        self.logger.info(post_fix)

        return metrics_values, str(post_fix)

    def check_data_coherence(file_path, user_ids, input_ids, answers, max_length=50):
        """
        Checks whether the data in the tensors is coherent with the data in the .txt file,
        padding sequences to max_length by adding zeros at the beginning.

        Args:
            file_path (str): Path to the .txt file containing the data.
            user_ids (torch.Tensor): Tensor of user IDs.
            input_ids (torch.Tensor): Tensor of input sequences (padded to max_length).
            answers (torch.Tensor): Tensor of answers (last item in each sequence).
            max_length (int): The maximum sequence length for padding. Default is 50.

        Returns:
            bool: True if data is coherent, False otherwise.
        """
        # Read and parse the .txt file
        data = {}
        with open(file_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue  # Skip empty lines
                user_id = int(tokens[0])
                item_ids = list(map(int, tokens[1:]))
                data[user_id] = item_ids

        # Check coherence
        for idx in range(len(user_ids)):
            user_id = user_ids[idx].item() if isinstance(user_ids[idx], torch.Tensor) else user_ids[idx]
            input_seq = input_ids[idx].tolist() if isinstance(input_ids[idx], torch.Tensor) else input_ids[idx]
            answer = answers[idx].item() if isinstance(answers[idx], torch.Tensor) else answers[idx]

            # Get the corresponding sequence from the data
            if user_id not in data:
                print(f"User ID {user_id} at index {idx} not found in the data.")
                return False

            sequence = data[user_id]

            # Extract the expected input sequence and answer
            expected_input_seq = sequence[:-1]
            expected_answer = sequence[-1]

            # Pad the expected input sequence to max_length by pre-padding with zeros
            expected_input_seq_padded = [0] * (max_length - len(expected_input_seq)) + expected_input_seq

            # Ensure the padded sequence is of length max_length
            if len(expected_input_seq_padded) > max_length:
                # If the sequence is longer than max_length, truncate from the beginning
                expected_input_seq_padded = expected_input_seq_padded[-max_length:]

            if input_seq != expected_input_seq_padded:
                print(f"Input sequence mismatch for user ID {user_id} at index {idx}.")
                print(f"Expected input sequence: {expected_input_seq_padded}")
                print(f"Actual input sequence:   {input_seq}")
                return False

            if answer != expected_answer:
                print(f"Answer mismatch for user ID {user_id} at index {idx}.")
                print(f"Expected answer: {expected_answer}")
                print(f"Actual answer:   {answer}")
                return False

        print("All user IDs, input sequences, and answers match the data.")
        return True
