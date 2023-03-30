from typing import Tuple
import logging
import math
import tenseal as ts


class LoggingUtils:

    def __init__(self, logger_name=None, log_level=logging.DEBUG):
        self.logger = logging.getLogger(logger_name)
        self.level = log_level
        self.logger.setLevel(log_level)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def add_file_handler(self, filename, level=logging.WARNING):
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(level)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    def add_console_handler(self, level=None):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level if level is None else level)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
