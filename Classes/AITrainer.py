from torch import LongTensor, Tensor
from Classes.MemoryManager import Memory_Manager
from Classes.TextEmbedder import Text_Embedder
from Classes.FilterModel import Filter_Model
from Classes.DictionaryService import Dictionary_Service
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
from threading import Thread
from time import sleep
from numpy.typing import NDArray
from numpy import float64, float32, argmax, intp
from numpy.random import randint


class Trainer:
    """
    It manages the training and interaction with the AI model.

    Attributes:
        __memory (Memory_Manager): The memory manager used to store and retrieve conversation data.
        __embedder (Text_Embedder): The text embedder used to convert text into embeddings.
        __filter_model (Filter_Model): The filter model used to filter embeddings.
        __dictionary (Dictionary_Service): The dictionary service used to store and retrieve definitions of words.
        __is_smart_active (bool): A boolean value that indicates whether the AI is smart or not.
        __passive_replies (List[str]): A list of passive replies that the AI can use when it is not smart.
        __tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The tokenizer used to tokenize input text.
        __model (AutoModelForCausalLM): The model used to generate responses.

    Methods:
        embedText(texts: List[str]) -> NDArray[float64]: Embedding the given texts unto numerical vectors.
        store(user_input: str, bot_response: str) -> None: Storing the user input and bot response in the memory.
        defineWord(word: str) -> str: Defining a word and storing the definition in the memory.
        retrieveMemory(query: str, key: int) -> str: Retrieving a given number of most similar questions from memory given a query.
        smartReply(user_input: str) -> str: Generating a response based on the user's input.
        getBestResponse(query: str) -> str: Retrieving the best response from memory given a query.
        reinforcement(query: str, response: str) -> None: Reinforcing the AI by providing a correct response to a question.
        trainAllWords() -> None: Training all words in the dictionary.
    """
    __memory: Memory_Manager
    __embedder: Text_Embedder
    __filter_model: Filter_Model
    __dictionary: Dictionary_Service
    __is_smart_active: bool
    __passive_replies: List[str]
    __tokenizer: AutoTokenizer
    __model: AutoModelForCausalLM

    def __init__(self):
        """
        Initializing the trainer with default values and starting the training for all words in a separate thread.
        """
        self.memory = Memory_Manager()
        self.embedder = Text_Embedder()
        self.filter_model = Filter_Model()
        self.dictionary = Dictionary_Service()
        self.is_smart_active = True
        self.passive_replies = ["Do you have a question?", "I'm still thinking about language...", "Define a word for me."]
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium") # type: ignore
        Thread(
            target=self.trainAllWords,
            daemon=True
        ).start()

    @property
    def memory(self) -> Memory_Manager:
        return self.__memory

    @memory.setter
    def memory(self, value: Memory_Manager) -> None:
        self.__memory = value

    @property
    def embedder(self) -> Text_Embedder:
        return self.__embedder

    @embedder.setter
    def embedder(self, value: Text_Embedder) -> None:
        self.__embedder = value

    @property
    def filter_model(self) -> Filter_Model:
        return self.__filter_model

    @filter_model.setter
    def filter_model(self, value: Filter_Model) -> None:
        self.__filter_model = value

    @property
    def dictionary(self) -> Dictionary_Service:
        return self.__dictionary

    @dictionary.setter
    def dictionary(self, value: Dictionary_Service) -> None:
        self.__dictionary = value

    @property
    def is_smart_active(self) -> bool:
        return self.__is_smart_active

    @is_smart_active.setter
    def is_smart_active(self, value: bool) -> None:
        self.__is_smart_active = value

    @property
    def passive_replies(self) -> List[str]:
        return self.__passive_replies

    @passive_replies.setter
    def passive_replies(self, value: List[str]) -> None:
        self.__passive_replies = value

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self.__tokenizer

    @tokenizer.setter
    def tokenizer(self, value: AutoTokenizer) -> None:
        self.__tokenizer = value

    @property
    def model(self) -> AutoModelForCausalLM:
        return self.__model

    @model.setter
    def model(self, value: AutoModelForCausalLM) -> None:
        self.__model = value

    def embedText(self, texts: List[str]) -> NDArray[float64]:
        """
        Embedding the given texts unto numerical vectors.

        Args:
            texts (List[str]): The texts to be embedded.

        Returns:
            NDArray[float64]: The embedded vectors.
        """
        return self.embedder.embed(texts)

    def store(self, user_input: str, bot_response: str) -> None:
        """
        Storing the user input and bot response in the memory.
        
        Args:
            user_input (str): The user's input.
            bot_response (str): The bot's response.
        """
        self.memory.store(user_input, bot_response)

    def defineWord(self, word: str) -> str:
        """
        Defining a word and storing the definition in the memory.

        Args:
            word (str): The word to be defined.

        Returns:
            str: The definition of the word.
        """
        return self.dictionary.defineWord(word)

    def retrieveMemory(
        self,
        query: str,
        key: int = 3
    ) -> str:
        """
        Retrieving a given number of most similar questions from memory given a query.

        Args:
            query (str): The query to search the memory for.
            key (int, optional): The number of most similar questions to be retrieved. Defaults to 3.

        Returns:
            str: A string containing the most similar questions and their respective answers.
        """
        return self.memory.retrieveMemory(query, self.embedText, key)

    def smartReply(self, user_input: str) -> str:
        """
        Generating a smart reply for a given user input.

        This method does th following:
            1. Checks if the smart reply feature is active.
            2. Checks if the user input is empty.
            3. Checks if the user input starts with "define " to define a word.
            4. Retrieves the most similar questions from memory.
            5. Generates a reply using the DialoGPT model.
            6. Stores the user input and reply in the memory.
            7. Returns the generated reply.

        Args:
            user_input (str): The user's input.

        Returns:
            str: The smart reply.
        """
        if not user_input.strip():
            return self.passive_replies[randint(len(self.passive_replies))]
        text: str = user_input.strip()
        if text.lower().startswith("define "):
            word: str = text.split("define ",1)[1]
            response: str = self.defineWord(word)
            self.store(text, response)
            return response
        memory_context: str = self.retrieveMemory(text)
        history: str = " ".join(f"User:{user_input} Bot:{bot_response}" for user_input, bot_response in self.memory.history[-3:])
        prompt: str = f"{memory_context} {history} User:{text} Bot:"
        inputs: BatchEncoding = self.tokenizer.encode_plus( # type: ignore
            text=prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        input_ids: Optional[Tensor] = inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("Input IDs are missing from the inputs dictionary")
        generated_output: LongTensor = self.model.generate( # type: ignore
            inputs=input_ids,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            pad_token_id=self.model.config.eos_token_id, # type: ignore
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        ) # type: ignore
        tokens: Tensor = generated_output[:, input_ids.shape[-1]:][0]
        reply: str = self.tokenizer.decode( # type: ignore
            token_ids=tokens,
            skip_special_tokens=True
        ).strip()
        if not reply:
            reply = "Could you elaborate?"
        self.store(text, reply)
        return reply

    def getBestResponse(self, query: str) -> str:
        """
        Retrieving the best response from memory given a query.
        
        Args:
            query (str): The query to search the memory for.
        
        Returns:
            str: The best response retrieved from memory.
        """
        if not self.memory.questions:
            return "No memory yet."
        texts: List[str] = self.memory.questions + [query]
        embed: NDArray[float64] = self.embedText(texts)
        scores: NDArray[float32] = self.filter_model.score(embed[:-1])
        best: intp = argmax(scores)
        return self.memory.answers[best]

    def reinforcement(self, query: str, response: str) -> None:
        """
        Reinforcing the AI by providing a correct response to a question.  It stores the question and response in the memory and trains the filter model with the new data.

        Args:
            query (str): The query to store in the memory.
            response (str): The correct response to the query.
        """
        self.store(query, response)
        if len(self.memory.questions) >= 4:
            inputs: List[str] = [f"{question} {answer}" for question, answer in zip(self.memory.questions, self.memory.answers)]
            labels: List[int] = [1 if "?" not in answer else 0 for answer in self.memory.answers]
            embedded: NDArray[float64] = self.embedText(inputs)
            self.filter_model.trainFilter(embedded, labels)

    def trainAllWords(self) -> None:
        """
        Training all words in the dictionary.

        This method does the following:
            1. Sleeps for 2 seconds.
            2. Retrieves the first 5000 words from the dictionary.
            3. For each word, defines it and stores the definition in the memory.

        Returns:
            (None)
        """
        sleep(2)
        all_words: List[str] = list(self.dictionary.dictionary.meaning.keys())[:5000] # type: ignore
        for index, word in enumerate(all_words):
            definitions: str = self.defineWord(word)
            self.store(f"define {word}", definitions)
