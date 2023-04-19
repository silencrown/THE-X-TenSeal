import os
import configparser

class ConfigUtils():
    def __init__(self, config_file=None):
        self.config = configparser.ConfigParser()
        if config_file is None or not os.path.exists(config_file):
            raise ValueError(f"Config file {config_file} is not exist.")
        self.config.read(config_file)
        self.config_dict = {}
        self._read_config()
    
    def __call__(self):
        return self.config_dict
    
    def _read_config(self):
        # logging module
        self.config_dict['logging_level'] = self.config.get('logging', 'level')

        # softmax_approx module
        self.config_dict['softmax_approx'] = self.config.get('model', 'softmax_approx')

# config file is located in the root directory of the project (thex's parent directory)
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini')
configer = ConfigUtils(CONFIG_FILE)
