import pandas as pd
import numpy as np
import torch


def split_user_group(uid_in_interaction):
    # 转为df结构
    u = uid_in_interaction.cpu().numpy()
    df = pd.DataFrame(u, columns=['uid'], dtype=np.int32)
    # 统计频次 & 降序排列
    df['inter_num'] = 0
    inter_num_res = df['uid'].value_counts()
    df['inter_num'] = df['uid'].map(inter_num_res)
    df.sort_values(by=['inter_num', 'uid'], ascending=[False, True], inplace=True)
    df.drop_duplicates(inplace=True)
    df.index = range(len(df))
    #print('df', df)

    # 取top5% & 名单
    all_num = df['uid'].nunique()
    active_num = int(all_num * 0.05)  # 47
    active_list = df.loc[:active_num, 'uid']
    inactive_list = df.loc[active_num:, 'uid']
    #print('active_list',active_list)
    #print('inactive_list',inactive_list)

    # return active_user_list,inactive_user_list


a = torch.Tensor([1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6])
split_user_group(a)
