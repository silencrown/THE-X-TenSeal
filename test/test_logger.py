
import test_helper
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