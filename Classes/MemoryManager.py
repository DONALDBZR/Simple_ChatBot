from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Callable
from numpy import argsort, float64, intp
from numpy.typing import NDArray


class Memory_Manager:
    """
    It stores and retrieves the conversation history between a user and a bot.

    Attributes:
        __history (List[Tuple[str, str]]): A list of tuples containing the user's input and the bot's response.
        __questions (List[str]): A list of user inputs.
        __answers (List[str]): A list of bot responses.

    Methods:
        store(user_input: str, bot_response: str) -> None: Storing the user input and bot response in the memory.
        retrieveMemory(query: str, embed_function: Callable[[List[str]], NDArray[float64]], key: int = 3) -> str: Retrieving a given number of most similar questions from memory given a query.
    """
    __history: List[Tuple[str, str]]
    __questions: List[str]
    __answers: List[str]

    def __init__(self):
        """
        Initializing the `Memory_Manager` with empty lists for history, questions and answers.
        """
        self.history = []
        self.questions = []
        self.answers = []

    @property
    def history(self) -> List[Tuple[str, str]]:
        return self.__history

    @history.setter
    def history(self, value: List[Tuple[str, str]]) -> None:
        self.__history = value

    @property
    def questions(self) -> List[str]:
        return self.__questions

    @questions.setter
    def questions(self, value: List[str]) -> None:
        self.__questions = value

    @property
    def answers(self) -> List[str]:
        return self.__answers

    @answers.setter
    def answers(self, value: List[str]) -> None:
        self.__answers = value

    def store(self, user_input: str, bot_response: str) -> None:
        """
        Storing the user input and bot response in the memory.
        
        Args:
            user_input (str): The user's input.
            bot_response (str): The bot's response.
        """
        self.history.append((user_input, bot_response))
        self.questions.append(user_input)
        self.answers.append(bot_response)

    def retrieveMemory(
        self,
        query: str,
        embed_function: Callable[[List[str]], NDArray[float64]],
        key: int = 3
    ) -> str:
        """
        Retrieving a given number of most similar questions from memory given a query.

        Args:
            query (str): The query to search the memory for.
            embed_function (Callable[[List[str]], NDArray[float64]]): A function that takes a list of strings and returns an embedding matrix.
            key (int, optional): The number of most similar questions to be retrieved. Defaults to 3.

        Returns:
            str: A string containing the most similar questions and their respective answers.
        """
        if not self.questions:
            return ""
        texts: List[str] = self.questions + [query]
        embeds: NDArray[float64] = embed_function(texts)
        similarity: NDArray[float64] = cosine_similarity(embeds[-1].reshape(1, -1), embeds[:-1])[0]
        indices: NDArray[intp] = argsort(similarity)[::-1][:key]
        return " ".join(f"User: {self.questions[index]} - Bot: {self.answers[index]}" for index in indices)
