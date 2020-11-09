#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-08-13

import logging
import os
import shutil

_a = set()


def init_logger(logger, log_folder=None, level='INFO', name="__name__"):

    if name in _a:
        return logger
    else:
        _a.add(name)

    if level == 'INFO':
        level = logging.INFO
    elif level == 'DEBUG':
        level = logging.DEBUG
    elif level == 'ERROR':
        level = logging.ERROR
    elif level == 'CRITICAL':
        level = logging.CRITICAL

    if log_folder is None:
        log_folder = "my_log"
        # if not os.path.isdir(self.log_folder):
        #     raise Exception(" %s 已经存在并且无法作为目录写入" % self.log_folder)
    else:
        log_folder = log_folder

    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)
    os.makedirs(log_folder)

    logger.setLevel(level=level)

    if not os.path.exists(os.path.join(log_folder, "log.txt")):
        # print(self.log_folder+"/log.txt")
        os.chdir(log_folder)
        f = open("log.txt", 'w')
        f.close()
        os.chdir("..")
        os.chdir("..")

    # init logger
    handler = logging.FileHandler(os.path.join(log_folder, "log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s %(module)s.%(funcName)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    return logger
