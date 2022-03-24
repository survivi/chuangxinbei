import os
import pandas as pd
import logging
import numpy as np

RANK_FILE = 'rank.csv'  # format: User_id \t [ranked item_ids] \t [scores] \t [labels]
GROUP_1_FILE = 'active_list.txt'
GROUP_2_FILE = 'inactive_list.txt'


class DataLoader(object):
    def __init__(self, path, sep='\t', seq_sep=',', label='label', rank_file=RANK_FILE, group_1_file=GROUP_1_FILE,
                 group_2_file=GROUP_2_FILE):
        self.rank_df = None
        self.path = path
        self.sep = sep
        self.seq_sep = seq_sep
        self.label = label
        self.rank_file = rank_file
        self.group_1_file = group_1_file
        self.group_2_file = group_2_file
        self._load_data()
        self.g1_df, self.g2_df = self._load_groups()

    def _load_data(self):
        rank_file = os.path.join(self.path, self.rank_file)
        if os.path.exists(rank_file):
            if self.rank_df is None:
                logging.info("load rank csv...")
                self.rank_df = pd.read_csv(rank_file, sep='\t')
                self.rank_df['q'] = 1
                # print("rank_df", self.rank_df)
                if 'uid' not in self.rank_df:
                    raise ValueError("missing uid in header.")
                logging.info("size of rank file: %d" % len(self.rank_df))
        else:
            raise FileNotFoundError('No rank file found.')

    def _load_groups(self):
        """
        Load advantaged/disadvantaged group info file and split the all data dataframe
        into two group-dataframes
        :return: group 1 dataframe (advantaged), group 2 dataframe (disadvantaged)
        """
        if self.rank_df is None:
            self._load_data()

        # print(self.rank_df)
        group_1_file = os.path.join(self.path, self.group_1_file)
        group_2_file = os.path.join(self.path, self.group_2_file)

        if os.path.exists(group_1_file):
            g1_user_arr = np.loadtxt(group_1_file, dtype=np.int32)
            g1_user_list = g1_user_arr.tolist()
        else:
            raise FileNotFoundError('No Group 1 file found.')

        if os.path.exists(group_2_file):
            g2_user_arr = np.loadtxt(group_2_file, dtype=np.int32)
            g2_user_list = g2_user_arr.tolist()
        else:
            raise FileNotFoundError('No Group 2 file found.')

        # print('g1_user_list',g1_user_list)
        # print('g2_user_list',g2_user_list)

        # 将总的排序列表中的user分组
        group_1_df = self.rank_df[self.rank_df['uid'].isin(g1_user_list)]
        group_2_df = self.rank_df[self.rank_df['uid'].isin(g2_user_list)]
        # print("group_1_df:-----------",g1_user_arr)
        return group_1_df, group_2_df
