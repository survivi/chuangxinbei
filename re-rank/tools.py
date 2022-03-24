import logging
from rank_metrics import *


def create_logger(name='result_logger', path='results.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(path)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def evaluation_methods(df, metrics):
    """
    Generate evaluation scores
    :param df:
    :param metrics:
    :return:
    """
    evaluations = []
    data_df = df.copy(deep=True)
    # q（0/1）表示是否从N个商品中选择该商品，作为top-k个推荐商品之一
    # 优化前q全为1，因此根据score排序取topk；优化后q有的为0，根据score*q排序取topk
    data_df["q*s"] = data_df['q'] * data_df['score']
    for metric in metrics:
        k = int(metric.split('@')[-1])
        tmp_df = data_df.sort_values(by='q*s', ascending=False, ignore_index=True)  # 降序
        # print("tmp_df", tmp_df)
        df_group = tmp_df.groupby('uid')  # 相同user的数据靠在一起
        # print("df_group", df_group)
        if metric.startswith('ndcg@'):
            ndcgs = []
            for uid, group in df_group:
                # print("uid", uid)
                # print("group", group)
                # print(group['label'].tolist()[:k])  #[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 推荐列表*user_num
                ndcgs.append(ndcg_at_k(group['label'].tolist()[:k], k=k, method=1))
            evaluations.append(np.average(ndcgs))
        elif metric.startswith('hit@'):
            hits = []
            for uid, group in df_group:
                hits.append(int(np.sum(group['label'][:k]) > 0))
            evaluations.append(np.average(hits))
        elif metric.startswith('precision@'):
            precisions = []
            for uid, group in df_group:
                if len(group['label'].tolist()) < k:
                    print(group)
                    print(uid)
                precisions.append(precision_at_k(group['label'].tolist()[:k], k=k))
            evaluations.append(np.average(precisions))
        elif metric.startswith('recall@'):
            recalls = []
            for uid, group in df_group:
                if np.sum(group['label']) == 0:
                    continue
                # 推荐的topk物品中在测试集中的物品占测试集物品总数的比例
                recalls.append(1.0 * np.sum(group['label'][:k]) / np.sum(group['label']))
            evaluations.append(np.average(recalls))
        elif metric.startswith('f1@'):
            f1 = []
            for uid, group in df_group:
                if np.sum(group['label']) == 0:
                    continue
                # print(np.sum(group['label']))
                f1.append(2 * np.sum(group['label'][:k]) / (np.sum(group['label']) + k))
            evaluations.append(np.average(f1))
    return evaluations
