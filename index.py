from Classes.Logger import ChatBot_Logger
from Classes.AITrainer import Trainer
from Classes.BotRunner import Bot_Runner
import warnings
from typing import Optional, TextIO, Union


logger: ChatBot_Logger = ChatBot_Logger(__name__)

def warningToLogger(
    message: Union[Warning, str],
    category: type[Warning],
    filename: str,
    line_number: int,
    file: Optional[TextIO] = None,
    line: Optional[str] = None
) -> None:
    log_message = f"{category.__name__}: {message} (File: {filename}, Line: {line_number})"
    logger.warning(log_message)

warnings.showwarning = warningToLogger
artificial_intelligence: Trainer = Trainer()
bot: Bot_Runner = Bot_Runner(artificial_intelligence, logger)
bot.start()
