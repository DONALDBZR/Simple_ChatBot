from Classes.Logger import ChatBot_Logger
from Classes.AITrainer import Trainer
from warnings import showwarning
from typing import Optional, TextIO, Union
from time import sleep
from numpy.random import randint
from threading import Thread


logger: ChatBot_Logger = ChatBot_Logger(__name__)

def warningToLogger(
    message: Union[Warning, str],
    category: type[Warning],
    filename: str,
    line_number: int,
    file: Optional[TextIO] = None,
    line: Optional[str] = None
) -> None:
    """
    A function to be passed to warnings.showwarning() to send warning messages to the logger.  This allows the AI to log warnings in a structured format, rather than printing them to the console.

    Args:
        message: The warning message.
        category: The category of the warning.
        filename: The name of the file where the warning occurred.
        line_number: The line number of the warning.
        file: The file object where the warning occurred. Defaults to None.
        line: The line of source code where the warning occurred. Defaults to None.
    """
    log_message: str = f"{category.__name__}: {message} (File: {filename}, Line: {line_number})"
    logger.warning(log_message)

showwarning = warningToLogger
artificial_intelligence: Trainer = Trainer()

def ai_input_loop() -> None:
    """
    A loop that takes user input and sends it to the AI for processing.  This allows users to interact with the AI in the console.

    The AI can be asked to train on new data by prefixing the input with "train:", and to recall a previous response by prefixing the input with "recall:".

    The loop will continue indefinitely until the user enters "quit" or "exit".

    Args:
        None

    Returns:
        None
    """
    while True:
        user_input: str = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        if user_input.startswith("train:"):
            text: str = user_input.split(":", 1)[1].strip()
            artificial_intelligence.reinforcement(text, "Trained.")
            print("AI: Training data added.")
            continue
        if user_input.startswith("recall:"):
            question: str = user_input.split(":", 1)[1].strip()
            print(f"AI: {artificial_intelligence.getBestResponse(question)}")
            continue
        print(f"AI: {artificial_intelligence.smartReply(user_input)}")

def ai_self_talker() -> None:
    """
    A loop that periodically sends empty strings to the AI's smartReply method to keep it "alive" and generate responses.  This allows the AI to generate responses without any user interaction.

    The loop will continue indefinitely until the program is stopped.

    Args:
        None

    Returns:
        None
    """
    while True:
        sleep(randint(8, 20))
        if artificial_intelligence.is_smart_active:
            reply: str = artificial_intelligence.smartReply("")
            print(f"AI: {reply}")

Thread(
    target=ai_input_loop,
    daemon=True
).start()
Thread(
    target=ai_self_talker,
    daemon=True
).start()
while True:
    sleep(1)
