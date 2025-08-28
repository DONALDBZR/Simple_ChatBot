from logging import Logger, getLogger, FileHandler, Formatter, DEBUG, Handler
from typing import List, Optional
from os import makedirs
from os.path import join


class Logger_Configurator:
    """
    It is a configurator for the logger with the settings specified.

    Attributes:
        __directory (str): The directory where the log file will be saved.
        __filename (str): The name of the log file.
        __format (str): The format of the log messages.
        __encoding (str): The encoding of the log file.
        __file_mode (str): The mode in which the log file will be opened.
        __handlers (List[Handler]): A list of handlers to be added to the logger.

    Methods:
        configure(self, logger_name: str) -> Logger: Configuring a logger with the given name and applies the specified settings.
    """
    __directory: str
    __filename: str
    __format: str
    __encoding: str
    __file_mode: str
    __handlers: List[Handler]

    def __init__(
        self,
        directory: str = "",
        filename: str = "ChatBot.log",
        format: str = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        encoding: str = "utf-8",
        file_mode: str = "a",
        handlers: Optional[List[Handler]] = None
    ):
        """
        Initializing the Logger_Configurator instance.

        This constructor does the following:
            1. Creates an instance of the `Logger_Configurator` with the given parameters.
            2. Sets the directory, filename, format, encoding, file mode and handlers of the logger configuration.
            3. If not specified, it will default to "./Logs" for the directory, "ChatBot.log" for the filename, "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s" for the format, "utf-8" for the encoding, "a" for the file mode and an empty list for the handlers.

        Args:
            directory (str): The directory where the log file will be stored.
            filename (str): The name of the log file.
            format (str): The format of the log messages.
            encoding (str): The encoding of the log file.
            file_mode (str): The mode of the log file.
            handlers (Optional[List[Handler]]): A list of handlers to use for the logger.
        """
        self.directory = directory if directory else "./Logs"
        self.filename = filename
        self.format = format
        self.encoding = encoding
        self.file_mode = file_mode
        self.handlers = handlers or []
        makedirs(
            name=self.directory,
            exist_ok=True
        )

    @property
    def directory(self) -> str:
        return self.__directory

    @directory.setter
    def directory(self, directory: str) -> None:
        self.__directory = directory

    @property
    def filename(self) -> str:
        return self.__filename

    @filename.setter
    def filename(self, filename: str) -> None:
        self.__filename = filename

    @property
    def format(self) -> str:
        return self.__format

    @format.setter
    def format(self, format: str) -> None:
        self.__format = format

    @property
    def encoding(self) -> str:
        return self.__encoding

    @encoding.setter
    def encoding(self, encoding: str) -> None:
        self.__encoding = encoding

    @property
    def file_mode(self) -> str:
        return self.__file_mode

    @file_mode.setter
    def file_mode(self, file_mode: str) -> None:
        self.__file_mode = file_mode

    @property
    def handlers(self) -> List[Handler]:
        return self.__handlers

    @handlers.setter
    def handlers(self, handlers: List[Handler]) -> None:
        self.__handlers = handlers

    def configure(self, logger_name: str) -> Logger:
        """
        Configuring a logger with the given name and applies the specified settings.

        Args:
            logger_name (str): The name of the logger to configure.

        Returns:
            Logger: The configured logger instance.
        """
        logger: Logger = getLogger(logger_name)
        logger.setLevel(DEBUG)
        if logger.hasHandlers():
            return logger
        file_handler: FileHandler = FileHandler(
            join(
                self.directory,
                self.filename
            ),
            mode=self.file_mode,
            encoding=self.encoding
        )
        formatter: Formatter = Formatter(self.format)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        for handler in self.handlers:
            logger.addHandler(handler)
        return logger
