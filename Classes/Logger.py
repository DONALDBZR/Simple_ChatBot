from logging.__init__ import Logger
from typing import Optional
from Classes.LoggerConfigurator import Logger_Configurator


class ChatBot_Logger:
    """
    It handles the logging for the application.

    Attributes:
        __logger (Logger): The logger instance.
        __configurator (Logger_Configurator): The configurator for the logger.

    Methods:
        debug(message: str) -> None: Logging the given message as a debug message.
        info(message: str) -> None: Logging the given message as information.
        warning(message: str) -> None: Logging the given message as a warning.
        error(message: str) -> None: Logging the given message as an error.
    """
    __logger: Logger
    __configurator: Logger_Configurator

    def __init__(
        self,
        name: str,
        configurator: Optional[Logger_Configurator] = None
    ):
        """
        Initializing the Extractio_Logger instance.

        This constructor does the following:
            1. Creates an instance of the `Extractio_Logger` with the given parameters.
            2. Sets the logger configurator to the given configurator if any, otherwise it will default to `Logger_Configurator()`.
            3. Sets the logger to the configured logger with the given name.

        Args:
            name (str): The name of the logger.
            configurator (Optional[Logger_Configurator], optional): The logger configurator. Defaults to None.
        """
        self.configurator = configurator or Logger_Configurator()
        self.logger = self.configurator.configure(name)

    @property
    def logger(self) -> Logger:
        return self.__logger

    @logger.setter
    def logger(self, logger: Logger) -> None:
        self.__logger = logger

    @property
    def configurator(self) -> Logger_Configurator:
        return self.__configurator

    @configurator.setter
    def configurator(self, configurator: Logger_Configurator) -> None:
        self.__configurator = configurator

    def debug(self, message: str) -> None:
        """
        Logging the given message as a debug message.

        Args:
            message (str): The message to log as a debug message.
        """
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """
        Logging the given message as information.

        Args:
            message (str): The message to log as information.
        """
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """
        Logging the given message as a warning.

        Args:
            message (str): The message to log as a warning.
        """
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """
        Logging the given message as an error.

        Args:
            message (str): The message to log as an error.
        """
        self.logger.error(message)
