from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Callable, Any
from numpy import ndarray, argsort


class Memory_Manager:
    """
    It stores and retrieves the conversation history between a user and a bot.

    Attributes:
        __history (List[Tuple[str, str]]): A list of tuples containing the user's input and the bot's response.
        __questions (List[str]): A list of user inputs.
        __answers (List[str]): A list of bot responses.

    Methods:
        store(user_input: str, bot_response: str) -> None: Storing the user input and bot response in the memory.
        retrieve_memory(query: str, embed_function: Callable[[List[str]], ndarray], key: int = 3) -> str: Retrieving a given number of most similar questions from memory given a query.
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

    def retrieve_memory(
        self,
        query: str,
        embed_function: Callable[[List[str]], ndarray],
        key: int = 3
    ) -> str:
        """
        Retrieving a given number of most similar questions from memory given a query.
        
        Args:
            query (str): The query to find similar questions.
            embed_function (Callable[[List[str]], ndarray]): The function to embed the questions.
            key (int, optional): The number of questions to retrieve. Defaults to 3.
        
        Returns:
            str: The retrieved questions.
        """
        if not self.questions:
            return ""
        texts: List[str] = self.questions + [query]
        embeds: ndarray = embed_function(texts)
        similarity: Any = cosine_similarity([embeds[-1]], embeds[:-1])[0]
        indices: ndarray = argsort(similarity)[::-1][:key]
        return " ".join(f"User: {self.questions[index]} - Bot: {self.answers[index]}" for index in indices)
