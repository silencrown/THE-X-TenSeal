import os
from datetime import datetime

import logging
import psutil
import configparser


CONFIG_FILE = "/home/gaosq/the-X-TenSeal/config.ini"

class LoggingUtils:

    def __init__(self, logger_name=None, config_file=None, file=False):       
        self.config = self._read_config(config_file)
        self.level = self.config.get('level')
        self.file = self.config.get('file')
        self.logger = logging.getLogger(logger_name)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
 
        self.logger.setLevel(self.level)
        self._add_console_handler()
        self._add_file_handler(self.file)


    def _read_config(self, config_file):
        if config_file is None:
            config_file = CONFIG_FILE

        config = configparser.ConfigParser()
        config.read(config_file)

        level_str = config.get('logging', 'level', fallback='info')
        print("level: {}".format(level_str))
        level = getattr(logging, level_str.upper())
        config_dict = {'level': level}

        file_path = config.get('logging', 'file', fallback=None)
        if file_path is not None:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            file_name = config.get('logging', 'file_name', fallback='the-x')
            config_dict.update({'file': os.path.join(file_path, file_name + self._get_data_time() + '.log')})
        return config_dict

    @staticmethod
    def _get_data_time():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _add_file_handler(self, filename):
        if filename is not None:
            file_handler = logging.FileHandler(filename)
            file_handler.setLevel(self.level)
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)

    def _add_console_handler(self):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
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
    
    @staticmethod
    def _bytes2human(n):
        """
        http://code.activestate.com/recipes/578019
        >>> bytes2human(10000)
        '9.8K'
        >>> bytes2human(100001221)
        '95.4M'
        """
        symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
        prefix = {}
        for i, s in enumerate(symbols):
            prefix[s] = 1 << (i + 1) * 10
        for s in reversed(symbols):
            if n >= prefix[s]:
                value = float(n) / prefix[s]
                return '%.1f%s' % (value, s)
        return "%sB" % n
    
    def log_system_info(self):
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        net_io_counters = psutil.net_io_counters()

        # log system info for debug
        self.logger.debug('CPU Usage: {}%'.format(cpu_percent))
        self.logger.debug('Memory Usage: {}%'.format(memory_info.percent))
        # self.logger.debug('Disk Usage: {}%'.format(disk_info.percent))
        # self.logger.debug('Network I/O: {} bytes sent, {} bytes received'.format(net_io_counters.bytes_sent, net_io_counters.bytes_recv))
    
        # moniter memory usage
        used_percent = memory_info.percent
        if used_percent > 80:
            self.logger.warning(f"Memory usage is over 80%: {used_percent}%")
            self.logger.warning(f"Memory usage: {self._bytes2human(memory_info.used)}")
        

    def __call__(self, message):
        """
        logging use default log level.
        """
        if self.level == logging.DEBUG:
            self.logger.debug(message)
        elif self.level == logging.INFO:
            self.logger.info(message)
        elif self.level == logging.WARNING:
            self.logger.warning(message)
        elif self.level == logging.ERROR:
            self.logger.error(message)

logger = LoggingUtils(logger_name="THE-X")