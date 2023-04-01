import logging
import configparser


CONFIG_FILE = "/home/gaosq/the-X-TenSeal/config.ini"

class LoggingUtils:

    def __init__(self, logger_name=None, config_file=None):
        self.logger = logging.getLogger(logger_name)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.level = self.read_config(config_file)
        self.logger.setLevel(self.level)

    def read_config(self, config_file):
        if config_file is None:
            config_file = CONFIG_FILE
        config = configparser.ConfigParser()
        config.read(config_file)
        level_str = config.get('logging', 'log_level')
        level = getattr(logging, level_str.upper())
        return level

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
    
    def __call__(self, message):
        if self.level == logging.DEBUG:
            self.logger.debug(message)
        elif self.level == logging.INFO:
            self.logger.info(message)
        elif self.level == logging.WARNING:
            self.logger.warning(message)
        elif self.level == logging.ERROR:
            self.logger.error(message)

logger = LoggingUtils(logger_name="THE-X")