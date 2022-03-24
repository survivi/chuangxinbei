from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_model, get_trainer, set_color
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='NeuMF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='beauty', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='basic.yaml amazon.yaml', help='config files')

    args, _ = parser.parse_known_args()
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    # configurations initialization
    config = Config(model=args.model, dataset=args.dataset, config_file_list=config_file_list)

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization & write config info into log
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)  # 经过筛选等操作之后的数据集（Dataset）
    logger.info(dataset)

    # dataset splitting  经过split之后的三部分（Dataloader）
    train_data, valid_data, test_data, g1_user_list, g2_user_list = data_preparation(config, dataset)
    # print('g1_user_list',g1_user_list)

    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'], fairfed_trainer=True)(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result, df_rank = trainer.evaluate_basic(test_data)  # evaluate_basic()!
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    #'''
    # rank_csv save 保存得分文件
    rank_save_path = './re-rank/data/{}/'.format(args.dataset)
    os.makedirs(rank_save_path, exist_ok=True)
    rank_file = args.model + '_rank_50.csv'
    df_rank.to_csv(rank_save_path + rank_file, columns=['uid', 'iid', 'score', 'label'], sep='\t', header=True, index=0)
    #'''

    # re-ranking
    # os.system('python re-rank.py ')



