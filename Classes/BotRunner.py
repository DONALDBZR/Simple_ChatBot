from Classes.AITrainer import Trainer
from Classes.Logger import ChatBot_Logger
from time import sleep
from numpy.random import randint
from threading import Thread


class Bot_Runner:
    """
    It manages the interation between the user and the AI Model.

    Attributes:
        __artificial_intelligence (Trainer): The AI model.
        __logger (ChatBot_Logger): The logger.
        __is_running (bool): The running status of the program.

    Methods:
        inputLoop(): A loop that takes user input and sends it to the AI for processing.
        start(): Starting the BotRunner's threads.
        selfTalkerLoop(): A loop that periodically sends empty strings to the AI's smartReply method to keep it "alive" and generate responses.
    """
    __artificial_intelligence: Trainer
    __logger: ChatBot_Logger
    __is_running: bool

    def __init__(self, artificial_intelligence: Trainer, logger: ChatBot_Logger):
        """
        Initializing the `Bot_Runner` instance with the given AI model and logger.

        This constructor does the following:
            1. Sets the AI model to the given AI model.
            2. Sets the logger to the given logger.
            3. Sets the running status to True.

        Args:
            artificial_intelligence (Trainer): The AI model.
            logger (ChatBot_Logger): The logger.
        """
        self.artificial_intelligence = artificial_intelligence
        self.logger = logger
        self.is_running = True

    @property
    def artificial_intelligence(self) -> Trainer:
        return self.__artificial_intelligence

    @artificial_intelligence.setter
    def artificial_intelligence(self, value: Trainer) -> None:
        self.__artificial_intelligence = value

    @property
    def logger(self) -> ChatBot_Logger:
        return self.__logger

    @logger.setter
    def logger(self, value: ChatBot_Logger) -> None:
        self.__logger = value

    @property
    def is_running(self) -> bool:
        return self.__is_running

    @is_running.setter
    def is_running(self, value: bool) -> None:
        self.__is_running = value

    def inputLoop(self):
        """
        A loop that takes user input and sends it to the AI for processing.  This allows users to interact with the AI in the console.

        The AI can be asked to train on new data by prefixing the input with "train:", and to recall a previous response by prefixing the input with "recall:".

        The loop will continue indefinitely until the user enters "quit" or "exit".

        Returns:
            None
        """
        while self.is_running:
            try:
                user_input: str = input("You: ")
            except (KeyboardInterrupt, EOFError):
                self.logger.error("User exited input loop.")
                break
            if user_input.lower() in ["quit", "exit"]:
                self.logger.info("User requested exit.")
                self.running = False
                break
            if user_input.startswith("train:"):
                text: str = user_input.split(":", 1)[1].strip()
                self.artificial_intelligence.reinforcement(text, "Trained.")
                print("AI: Training data added.")
                continue
            if user_input.startswith("recall:"):
                question: str = user_input.split(":", 1)[1].strip()
                print(f"AI: {self.artificial_intelligence.getBestResponse(question)}")
                continue
            print(f"AI: {self.artificial_intelligence.smartReply(user_input)}")

    def selfTalkerLoop(self):
        """
        A loop that periodically sends empty strings to the AI's smartReply method to keep it "alive" and generate responses.  This allows the AI to generate responses without any user interaction.

        The loop will continue indefinitely until the user stops the program.

        Returns:
            None
        """
        while self.is_running:
            sleep(randint(8, 20))
            if self.artificial_intelligence.is_smart_active:
                reply: str = self.artificial_intelligence.smartReply("")
                print(f"AI: {reply}")

    def start(self):
        """
        Starting the BotRunner's threads.

        This method starts two threads: one for the AI's input loop and one for the AI's self-talking loop.  The method will block until the user stops the program.

        Returns:
            None
        """
        self.logger.info("Starting BotRunner threads.")
        Thread(
            target=self.inputLoop,
            daemon=True
        ).start()
        Thread(
            target=self.selfTalkerLoop,
            daemon=True
        ).start()
        while self.running:
            sleep(1)
