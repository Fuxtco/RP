# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import logging
import time
from datetime import timedelta
import pandas as pd


class LogFormatter:
    # 格式化日志

    def __init__(self):
        self.start_time = time.time()

    # record: Python内置日志系统传入的一条
    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        # 构成日志前缀
        prefix = "%s - %s - %s" % (
            record.levelname, #INFO, DEBUG, ERROR
            time.strftime("%x %X"), #当前日期&时间
            timedelta(seconds=elapsed_seconds), #将秒数转换成时间格式
        )
        message = record.getMessage()
        # 把每个换行换成换行加前缀长度和三个空格的缩进，满足缩进格式，便于查看
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = "%s-%i" % (filepath, rank)
        # a-追加模式
        file_handler = logging.FileHandler(filepath, "a")
        # 记录所有级别目录
        file_handler.setLevel(logging.DEBUG)
        # 设置格式
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger() 
    logger.handlers = [] #避免重复挂载反复打印
    logger.setLevel(logging.DEBUG)
    logger.propagate = False #禁止日志向上层传播
    # 挂载两种句柄
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    # 便于查看每个epoch的时间
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger


class PD_Stats(object):
    """
    Log stuff with pandas library
    """

    def __init__(self, path, columns):
        self.path = path

        # reload path stats
        if os.path.isfile(self.path):
            self.stats = pd.read_pickle(self.path)

            # check that columns are the same
            assert list(self.stats.columns) == list(columns)

        else:
            self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row

        # save the statistics
        if save:
            self.stats.to_pickle(self.path)
