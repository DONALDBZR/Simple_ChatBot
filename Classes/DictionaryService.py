from nltk.corpus import wordnet
from pydictionary import Dictionary
from typing import List, Any, Optional


class DictionaryService:
    """
    It is the service that will act as dictionary.

    Attributes:
        dictionary: (Dictionary): The dictionary that will be used to find the definition of the given word.

    Methods:
        define_word(word: str) -> str: Retrieving the definition of the given word.
    """
    __dictionary: Dictionary

    def __init__(self):
        pass

    @property
    def dictionary(self) -> Dictionary:
        return self.__dictionary

    @dictionary.setter
    def dictionary(self, dictionary: Dictionary) -> None:
        self.__dictionary = dictionary

    def define_word(self, word: str) -> str:
        """
        Retrieving the definition of the given word.

        This method does the following:
            1. Checks if the word is found in PyDictionary.
            2. If not, it searches for synsets in WordNet.
            3. Returns the definition of the word or "No definition found." if it was not found.

        Args:
            word (str): The word to find the definition for.

        Returns:
            str: The definition of the word or "No definition found." if it was not found.
        """
        self.dictionary = Dictionary(word)
        definitions: Optional[List[Any]] = self.dictionary.meanings()
        if definitions:
            return '; '.join(f"{key}: {', '.join(value)}" for key, value in definitions.items()) # type: ignore
        synsets: List[Any] = wordnet.synsets(word)
        if synsets:
            return '; '.join(set(synset.definition() for synset in synsets))
        return "No definition found."
