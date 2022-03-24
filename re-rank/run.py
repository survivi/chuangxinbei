import os
import argparse

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from data_loader import DataLoader
from tools import *


class UGF(object):
    def __init__(self, data_loader, k, eval_metric_list, fairness_metric='f1',
                 epsilon=0, logger=None, model_name='', group_name=''):
        """
        Train fairness model
        :param data_loader: Dataloader object
        :param k: k for top-K number of items to be selected from the entire list
        :param eval_metric_list: a list contains all the metrics to report
        :param fairness_metric: a string, the metric used for fairness constraint, default='f1'
        :param epsilon: the upper bound for the difference between two groups scores
        :param logger: logger for logging info
        :param model_name: 基础推荐模型
        :param group_name: 分组方式
        """
        self.data_loader = data_loader
        self.dataset_name = data_loader.path.split('/')[-1]
        self.k = k
        self.eval_metric_list = eval_metric_list
        self.fairness_metric = fairness_metric
        self.epsilon = epsilon
        self.model_name = model_name
        self.group_name = group_name
        if logger is None:
            self.logger = create_logger()
        else:
            self.logger = logger

    @staticmethod
    def _check_df_format(df):
        """
        check if the input dataframe contains all the necessary columns
        :return: None
        """
        expected_columns = ['uid', 'iid', 'score', 'label', 'q']
        for c in expected_columns:
            if c not in df.columns:
                raise KeyError('Missing column ' + c)

    @staticmethod
    def _build_fairness_optimizer(group_df_list, k, metric, name='UGF'):
        """
        Use Gurobi to build faireness optimizer
        :param group_df_list: a list contains dataframes from two groups
        :param k: an integer for the length of top-K list
        :param metric: the metric string for fairness constraint. e.g. 'f1', 'recall', 'precision'
        :param name: a string which is the name of this optimizer
        :return: the Gurobi model, a list of Qi*Si for obj function, a list of two group metric
        """
        try:
            # Create a new model
            m = gp.Model(name)
            var_score_list = []  # store the Qi * Si
            metric_list = []  # store the averaged metric scores for two groups
            # Create variables
            for df in group_df_list:
                df_group = df.groupby('uid')
                tmp_metric_list = []  # store the metric calculation for each user in current user group
                tmp_var_score_list = []  # store var * score for object function use
                for uid, group in df_group:
                    # print(uid)
                    # print(group)
                    tmp_var_list = []  # store variables for sum(Qi) == k use
                    tmp_var_label_list = []  # store var * label for recall calculation use
                    score_list = group['score'].tolist()
                    label_list = group['label'].tolist()
                    item_list = group['iid'].tolist()

                    for i in range(len(item_list)):
                        var_name = str(uid) + '_' + str(item_list[i])  # variable name is "uid_iid"
                        v = m.addVar(vtype=GRB.BINARY, name=var_name)
                        # print(v,score_list[i] * v)
                        # print("v",score_list[i] * v)
                        tmp_var_list.append(v)
                        tmp_var_score_list.append(score_list[i] * v)
                        tmp_var_label_list.append(label_list[i] * v)
                    # Add first constraint: Sum(Qi)==k

                    m.addConstr(gp.quicksum(tmp_var_list) == k)
                    # calculate the corresponding measures 公平性指标UGF计算！
                    if group['label'].sum() == 0:
                        continue
                    if metric == 'recall':
                        tmp_metric_list.append(gp.quicksum(tmp_var_label_list) / group['label'].sum())
                    elif metric == 'precision':
                        tmp_metric_list.append(gp.quicksum(tmp_var_label_list) / k)
                    elif metric == 'f1':
                        f1 = 2 * gp.quicksum(tmp_var_label_list) / (group['label'].sum() + k)
                        tmp_metric_list.append(f1)
                    else:
                        raise ValueError('Unknown metric for optimizer building.')

                metric_list.append(gp.quicksum(tmp_metric_list) / len(tmp_metric_list))  # 两个组的指标值
                var_score_list.extend(tmp_var_score_list)  # 在var_score_list列表后扩充tmp_var_score_list的值
            m.update()
            return m, var_score_list, metric_list

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
        except AttributeError:
            print('Encountered an attribute error')
        pass

    @staticmethod
    def _format_result(model, df):
        """
        format the gurobi results to dataframe.
        :param model: optimized gurobi model
        :param df: the pandas dataframe to add the optimized results into
        :return: None
        """
        """
        print("model.getVars()",model.getVars()[0])
        for v in model.getVars():
            #print("v.varName",v.varName)
            #print("v.x",v.x)
            v_s = v.varName.split('_')
            uid = int(v_s[0])
            iid = int(v_s[1])
            df.loc[(df['uid'] == uid) & (df['iid'] == iid), 'q'] = int(v.x)
        """
        # print(model.getVars())
        new_df = pd.DataFrame(model.getVars(), columns=['uid_iid_q']).astype(str)
        # print(new_df)
        new_df['uid'] = new_df['uid_iid_q'].map(lambda x: x.split(" ")[1].split("_")[0]).astype(int)
        new_df['iid'] = new_df['uid_iid_q'].map(lambda x: x.split(" ")[1].split("_")[1]).astype(int)
        new_df['q'] = new_df['uid_iid_q'].map(lambda x: x.split(" ")[3].split(")")[0]).astype(float)
        # print("new_df", new_df)
        df = pd.merge(df, new_df, on=['uid', 'iid'], how='left')
        # print("df+new_df",df)
        # print(df["q_y"])
        return df['q_y']

    def _print_metrics(self, df, metrics, message='metric scores'):
        """
        Print out evaluation scores
        :param df: the dataframe contains the data for evaluation
        :param metrics: a list, contains the metrics to report
        :param message: a string, for print message
        :return: None
        """
        results = evaluation_methods(df, metrics=metrics)
        # print(results)
        r_string = ""
        for i in range(len(metrics)):
            r_string = r_string + metrics[i] + "=" + '{:.4f}'.format(results[i]) + " "
        # print(message + ": " + r_string)
        # write the message into the log file
        self.logger.info(message + ": " + r_string)
        return results

    def train(self):
        """
        Train fairness model
        """
        # model = read('gurobi_model.mps')

        # Prepare data
        all_df = self.data_loader.rank_df.copy(deep=True)  # the dataframe with entire test data
        # print('all_df',all_df)
        self._check_df_format(all_df)  # check the dataframe format
        group_df_list = [self.data_loader.g1_df.copy(deep=True),
                         self.data_loader.g2_df.copy(deep=True)]  # group 1 (active), group 2 (inactive)

        # Print original evaluation results
        print('step1')
        self.logger.info('Model:{} | Dataset:{} |  Epsilon={} | K={} | GRU_metric={}'
                         .format(self.model_name, self.dataset_name, self.epsilon, self.k, self.fairness_metric))
        result = self._print_metrics(all_df, self.eval_metric_list, 'Before optimization overall scores           ')
        result1 = self._print_metrics(group_df_list[0], self.eval_metric_list,
                                      'Before optimization group 1 (active) scores  ')
        result2 = self._print_metrics(group_df_list[1], self.eval_metric_list,
                                      'Before optimization group 2 (inactive) scores')
        self.logger.info("Before optimization ------------------ UFG-f1:{:.4f}".format(np.abs(result1[1] - result2[1])))
        print('\n')

        # print(group_df_list)
        # build optimizer
        m, var_score_list, metric_list = \
            self._build_fairness_optimizer(group_df_list, self.k, metric=self.fairness_metric, name='UGF_f1')
        # print('var_score_list',var_score_list)
        # print('metric_list',metric_list[0])

        # |group_1_f1 - group_2_f1| <= epsilon
        m.addConstr(metric_list[0] - metric_list[1] <= self.epsilon)
        m.addConstr(metric_list[1] - metric_list[0] <= self.epsilon)

        # Set objective function
        m.setObjective(gp.quicksum(var_score_list), GRB.MAXIMIZE)

        # Optimize model
        m.optimize()
        # m.write('gurobi_model.mps')

        # Format the output results and update q column of the dataframe
        # print("all_df 更新前", all_df)
        all_df['q'] = self._format_result(m, all_df)  # 更新后的q,即选不选该商品作为推荐物品，（最初所有的物品q均为1）
        # print("all_df 更新后", all_df)

        group_df_list[0].drop(columns=['q'], inplace=True)
        print("group_df_list[0]", group_df_list[0])
        # group[0]向all_df中获取新的"q"，q表示item是否被推荐
        group_df_list[0] = pd.merge(group_df_list[0], all_df, on=['uid', 'iid', 'score', 'label'], how='left')
        print(group_df_list[0])
        group_df_list[1].drop(columns=['q'], inplace=True)
        print("group_df_list[1]", group_df_list[1])
        group_df_list[1] = pd.merge(group_df_list[1], all_df, on=['uid', 'iid', 'score', 'label'], how='left')
        # Print updated evaluation results
        print('\n')
        print('step2')
        result = self._print_metrics(all_df, self.eval_metric_list, 'After optimization overall metric scores     ')
        result1 = self._print_metrics(group_df_list[0], self.eval_metric_list,
                                      'After optimization group 1 (active) scores   ')
        result2 = self._print_metrics(group_df_list[1], self.eval_metric_list,
                                      'After optimization group 2 (inactive) scores ')
        self.logger.info("After optimization ------------------- UFG-f1:{:.4f}".format(np.abs(result1[1] - result2[1])))

        self.logger.info('\n\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='beauty', help='name of datasets')
    parser.add_argument('--e', type=float, default=2, help='epsilon')  # 0~0.1
    parser.add_argument('--topk', type=list, default=['10'], help='topk')
    parser.add_argument('--neg_num', type=int, default=100, help='negative samples number')
    args, _ = parser.parse_known_args()

    logger_dir = os.path.join('results/', args.model)  # logging file path
    if not os.path.exists(logger_dir):
        os.mkdir(logger_dir)
    logger_file = args.model + '_' + args.dataset + '_reRank_result.log'
    logger_path = os.path.join(logger_dir, logger_file)
    logger = create_logger(name='result_logger', path=logger_path)

    dataset_folder = './data'  # dataset directory
    data_path = os.path.join(dataset_folder, args.dataset)
    file_name='_rank_{}.csv'.format(args.neg_num)
    rank_file = args.model + file_name  # original input ranking csv file name (注意修改！！)
    dl = DataLoader(data_path, rank_file=rank_file)

    metrics = ['ndcg', 'f1']
    metrics_list = [metric + '@' + k for metric in metrics for k in args.topk]

    UGF_model = UGF(dl, k=10, eval_metric_list=metrics_list, fairness_metric='f1',
                    epsilon=args.e, logger=logger, model_name=args.model)
    UGF_model.train()
