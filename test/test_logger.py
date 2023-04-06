import sys
import os

the_x_parent_directory = os.path.abspath("/home/gaosq/the-X-TenSeal/")
if the_x_parent_directory not in sys.path:
    sys.path.insert(0, the_x_parent_directory)

from thex._logger import logger


def test_logger():
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")
    logger.log_system_info()
    logger("test")

if __name__ == '__main__':
    test_logger()