import logging
from colorlog import ColoredFormatter

class LoggerUtils:
    def __init__(self):
        # 创建一个日志器
        self.logger = logging.getLogger('example_logger')
        self.logger.setLevel(logging.INFO)

        # 检查日志器是否已经有处理器，避免重复添加
        if not self.logger.hasHandlers():
            # 创建彩色日志格式
            log_format = "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s"
            formatter = ColoredFormatter(log_format)

            # 创建一个流处理器，并将其添加到日志器
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

    def info(self, msg):
        self.logger.info(str(msg))


if __name__ == "__main__":
    loggertool = LoggerUtils()
    loggertool.logger.info("test")