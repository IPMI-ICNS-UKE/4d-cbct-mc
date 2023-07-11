import logging


class LoggerMixin:
    @property
    def logger(self):
        try:
            self.__logger
        except AttributeError:
            logger_name = f"{self.__module__}.{self.__class__.__name__}"
            self.__logger = logging.getLogger(logger_name)
        return self.__logger
