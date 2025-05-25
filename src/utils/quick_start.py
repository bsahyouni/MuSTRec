"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
import time

from sympy.logic.inference import valid
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
from models.sequential_head import SequentialModel
from utils.seq_dataset import get_seq_dic, get_rating_matrix

from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.init as init
import platform
import os
import csv
import wandb


def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    # merge config dict
    config = Config(model, dataset, config_dict, mg)
    # config['epochs'] = 1
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    seq_dic, max_items, num_users = get_seq_dic()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    whatever = ""
    config['valid_rating_matrix'], config['test_rating_matrix'] = get_rating_matrix(whatever, seq_dic, max_items)

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, max_items, batch_size=config['train_batch_size'], shuffle=True, seq_dict=seq_dic)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, max_items, additional_dataset=train_dataset, batch_size=config['eval_batch_size'], seq_dict=seq_dic),
        EvalDataLoader(config, test_dataset, max_items, additional_dataset=train_dataset, batch_size=config['eval_batch_size'], seq_dict=seq_dic, valid=False))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    # Define the omega values to sweep over
    omega_values = [0.1]
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        for omega in omega_values:
            logger.info('========={}/{}: Parameters:{}={}======='.format(
                idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

            #embeddings outside of models to update jointly
            n_items = dataset.get_item_num()
            n_users = dataset.get_user_num()
            embedding_dim = config['embedding_size']

            item_embeddings = nn.Embedding(n_items, embedding_dim, padding_idx=0).to(config['device'])
            user_embeddings = nn.Embedding(n_users, embedding_dim).to(config['device'])

            init.xavier_uniform_(item_embeddings.weight)
            init.xavier_uniform_(user_embeddings.weight)

            # set random state of dataloader
            train_data.pretrain_setup()

            ablation_scenarios = [
                {
                    "label": "MuSTRec",
                    "changes": {
                        # No changes
                    }
                },
                # {
                #     "label": "BaseCase",
                #     "changes": {
                #         # Both turned off
                #         "multimodal_switchoff": 1
                #     }
                # },
                # {
                #     "label": "RegWeightOnly",
                #     "changes": {
                #         # Turn on reg_weight, keep ui_graph_weight off
                #         # and do NOT touch mm_image_weight or alpha
                #         "ui_graph_weight": 0.0,
                #         "reg_weight": 0.001  # Use "standard" non-zero reg_weight here
                #     }
                # },
                # {
                #     "label": "UIGraphOnly",
                #     "changes": {
                #         # Turn on ui_graph_weight, keep reg_weight off
                #         "ui_graph_weight": 1.0,  # Or a typical "on" value in your code
                #         "reg_weight": 0
                #     }
                # },
                # {
                #     "label": "RegWeightOn_mmImage0",
                #     "changes": {
                #         # Turn on reg_weight, mm_image_weight=0, keep ui_graph_weight=0
                #         "ui_graph_weight": 0.0,
                #         "reg_weight": 0.001,
                #         "mm_image_weight": 0.0
                #     }
                # },
                # {
                #     "label": "RegWeightOn_mmImage1",
                #     "changes": {
                #         # Turn on reg_weight, mm_image_weight=1, keep ui_graph_weight=0
                #         "ui_graph_weight": 0.0,
                #         "reg_weight": 0.001,
                #         "mm_image_weight": 1.0
                #     }
                # }
            ]

            # import and initialise seq model
            # required args for initialisation
            config_sequential = SimpleNamespace(
                data_dir="./data/",
                output_dir="output/",
                data_name="New_Sports_and_Outdoors",
                do_eval=False,  # since "store_true" means default is False
                load_model=None,
                train_name="get_local_time()",  # Adjust this to actually call the function or set a default name
                num_items=10,
                num_users=10,
                no_cuda=False,  # "store_true" so default is False
                log_freq=1,
                patience=10,
                num_workers=4,
                seed=42,
                weight_decay=0.0,
                adam_beta1=0.9,
                adam_beta2=0.999,
                gpu_id="0",
                variance=5,
                model_type='SequentialModel',
                max_seq_length=50,
                hidden_size=64,
                num_hidden_layers=2,
                hidden_act="gelu",
                num_attention_heads=2,
                attention_probs_dropout_prob=0.5,
                hidden_dropout_prob=0.5,
                initializer_range=0.02,
                pretrained='false',
                c=3,
                alpha=0.9,
                device = 'cuda'
            )

            for scenario in ablation_scenarios:
                logger.info(f"\n\n===== Running Ablation Scenario: {scenario['label']} =====")

                model = get_model(config['model'])(item_embeddings, user_embeddings, config, train_data).to(config['device'])
                seq_model = SequentialModel(config_sequential, item_embeddings, user_embeddings=None).to(config['device'])

                # trainer loading and initialization
                trainer = get_trainer()(config, model, model2=seq_model, mg=mg, user_embeddings=user_embeddings, item_embeddings=item_embeddings)
                if wandb.run is None:  # avoid creating duplicate runs in case you call quick_start twice
                    wandb.init(
                        project="MuSTRec",
                        name=f"{model}_{dataset}_{time.strftime('%Y%m%d-%H%M%S')}",
                        config=config_dict,  # or ConfigObj.to_dict() if you prefer
                    )
                trainer.omega = omega

                # Apply each parameter override in scenario['changes']:
                for param_name, new_value in scenario['changes'].items():
                    if param_name == "multimodal_switchoff":
                        trainer.multimodal_switchoff = new_value
                    elif param_name == "reg_weight":
                        trainer.model.reg_weight = new_value
                    elif param_name == "mm_image_weight":
                        trainer.model.mm_image_weight = new_value
                    elif param_name == "ui_graph_weight":
                        trainer.model.ui_graph_weight = new_value
                    # Add more as needed.

                # # ── Model-level statistics ────────────────────────────────
                # model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                # seq_model_params = sum(p.numel() for p in seq_model.parameters() if p.requires_grad)
                # total_params = model_params + seq_model_params
                #
                # # ── 1️⃣ Training epoch time ────────────────────────────────
                # t0 = time.perf_counter()
                # best_valid_seq, best_valid_result, best_test_seq, best_test_upon_valid = trainer.fit(
                #     train_data, valid_data=valid_data, test_data=test_data, saved=save_model
                # )
                # train_epoch_time = time.perf_counter() - t0
                #
                # # ── 2️⃣ Test epoch time (one full pass on test loader) ────
                # t1 = time.perf_counter()
                # try:  # evaluate() exists in every default RecBole trainer
                #     trainer.evaluate(test_data, phase='test')
                # except AttributeError:  # fall back gracefully if a custom trainer has no evaluate()
                #     pass
                # test_epoch_time = time.perf_counter() - t1
                #
                # # ── 3️⃣ Write / append the CSV row ─────────────────────────
                # stats_csv = "computation_stats.csv"
                # header = ["model_name", "total_params", "train_epoch_time", "test_epoch_time"]
                # write_hdr = not os.path.exists(stats_csv)
                #
                # with open(stats_csv, "a", newline="") as f:
                #     w = csv.writer(f)
                #     if write_hdr:
                #         w.writerow(header)
                #     w.writerow([scenario["label"], total_params,
                #                 f"{train_epoch_time:.4f}", f"{test_epoch_time:.4f}"])
                #
                # logger.info("Saved model stats to %s", stats_csv)
                #
                # # ── 4️⃣ Stop the script after collecting the first epoch’s info ─
                # return  # ⬅ this exits quick_start() cleanly
                # # ──────────────────────────────────────────────────────────

                # model training
                best_valid_seq, best_valid_result, best_test_seq, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
                #########
                hyper_ret.append((hyper_tuple, best_valid_seq, best_test_seq))

                # save best test
                if best_test_seq[12] > best_test_value:
                    best_test_value = best_test_seq[12]
                    best_test_idx = idx
                idx += 1

                logger.info('best valid result: {}'.format(dict2str(best_valid_seq)))
                logger.info('test result: {}'.format(dict2str(best_test_seq)))
                logger.info('████Current BEST████:\nParameters: {}={},\n'
                            'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
                    hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

                # Save results to CSV
                csv_file = 'test.csv'
                metrics = ['HR@5', 'NDCG@5', 'Precision@5', 'MAP@5',
                           'HR@10', 'NDCG@10', 'Precision@10', 'MAP@10',
                           'HR@15', 'NDCG@15', 'Precision@15', 'MAP@15',
                           'HR@20', 'NDCG@20', 'Precision@20', 'MAP@20']
                header = ['omega', 'scenario'] + ['valid_' + metric for metric in metrics] + ['test_' + metric for metric in metrics]
                # header = ['omega'] + ['valid_' + metric for metric in metrics] + ['test_' + metric for metric in
                #                                                                               metrics]

                # Write header if file doesn't exist
                file_exists = os.path.isfile(csv_file)
                if not file_exists:
                    with open(csv_file, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(header)

                # Collect data for CSV
                #row = [omega]
                row = [omega, scenario["label"]]
                row.extend(best_valid_seq)
                row.extend(best_test_seq)

                # Write the row to the CSV file
                with open(csv_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row)
                logger.info(f'Completed training for omega={omega}')

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))



