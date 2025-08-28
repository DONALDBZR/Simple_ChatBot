import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import keras
import sklearn
import xgboost as xgb
import scipy
import nltk
import spacy
import transformers
import cv2
import lightgbm as lgb
import catboost as cb
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
from PyDictionary import PyDictionary
import threading
import time

nltk.download('punkt')
nltk.download('wordnet')

try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
dictionary = PyDictionary()

class ResponseFilterNN(nn.Module):
    def __init__(self, input_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class AITrainer:
    def __init__(self):
        self.history = []
        self.memory_q = []
        self.memory_a = []
        self.vectorizer = TfidfVectorizer()
        self.filter_model = ResponseFilterNN()
        self.optimizer = optim.Adam(self.filter_model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()
        self.smart_active = True
        self.passive_replies = ["Do you have a question?", "I'm still thinking about language...", "Define a word for me."]
        threading.Thread(target=self.train_all_words, daemon=True).start()

    def embed_text(self, texts):
        vectors = self.vectorizer.fit_transform(texts)
        n_features = vectors.shape[1]
        n_comp = min(100, n_features)
        svd = TruncatedSVD(n_components=n_comp)
        reduced = svd.fit_transform(vectors)
        if n_comp < 100:
            pad = np.zeros((reduced.shape[0], 100 - n_comp))
            reduced = np.hstack((reduced, pad))
        return reduced

    def store(self, user_input, bot_response):
        self.history.append((user_input, bot_response))
        self.memory_q.append(user_input)
        self.memory_a.append(bot_response)

    def train_filter(self):
        if len(self.memory_q) < 4:
            return
        inputs = [q + " " + a for q, a in zip(self.memory_q, self.memory_a)]
        labels = [1 if "?" not in a else 0 for a in self.memory_a]
        embedded = self.embed_text(inputs)
        X = torch.tensor(embedded, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        self.filter_model.train()
        for _ in range(10):
            self.optimizer.zero_grad()
            out = self.filter_model(X)
            loss = self.loss_fn(out, y)
            loss.backward()
            self.optimizer.step()

    def define_word(self, word):
        defs = dictionary.meaning(word)
        if defs:
            return '; '.join(f"{k}: {', '.join(v)}" for k,v in defs.items())
        syns = wordnet.synsets(word)
        if syns:
            return '; '.join(set(s.definition() for s in syns))
        return "No definition found."

    def retrieve_memory(self, query, k=3):
        if not self.memory_q:
            return ""
        texts = self.memory_q + [query]
        embed = self.embed_text(texts)
        sim = cosine_similarity([embed[-1]], embed[:-1])[0]
        idxs = np.argsort(sim)[::-1][:k]
        return ' '.join(f"User:{self.memory_q[i]} Bot:{self.memory_a[i]}" for i in idxs)

    def smart_reply(self, user_input):
        if not user_input.strip():
            return self.passive_replies[np.random.randint(len(self.passive_replies))]
        text = user_input.strip()
        low = text.lower()
        if low.startswith("define "):
            word = text.split("define ",1)[1]
            resp = self.define_word(word)
            self.store(text, resp)
            return resp

        mem_ctx = self.retrieve_memory(text)
        hist = ' '.join(f"User:{u} Bot:{b}" for u,b in self.history[-3:])
        prompt = f"{mem_ctx} {hist} User:{text} Bot:"
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        gen = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )
        reply = tokenizer.decode(gen[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True).strip()
        if not reply:
            reply = "Could you elaborate?"
        self.store(text, reply)
        return reply

    def get_best_response(self, query):
        if not self.memory_q:
            return "No memory yet."
        texts = self.memory_q + [query]
        embed = self.embed_text(texts)
        X = torch.tensor(embed[:-1], dtype=torch.float32)
        self.filter_model.eval()
        with torch.no_grad():
            scores = self.filter_model(X)[:,1].numpy()
        best = np.argmax(scores)
        return self.memory_a[best]

    def reinforcement(self, q, r):
        self.memory_q.append(q)
        self.memory_a.append(r)
        self.train_filter()

    def train_all_words(self):
        time.sleep(2)
        print("[AI Boot] Starting language comprehension training...")
        all_words = list(wordnet.words())[:5000]
        for i, word in enumerate(all_words):
            defs = self.define_word(word)
            if isinstance(defs, str):
                self.store(f"define {word}", defs)
            elif isinstance(defs, list):
                self.store(f"define {word}", "; ".join(defs))
            if i % 250 == 0:
                print(f"[AI Boot] Trained on {i} words...")
        print("[AI Boot] Training complete.")

def ai_input_loop():
    while True:
        u = input("You: ")
        if u.lower() in ['quit','exit']:
            break
        if u.startswith('train:'):
            txt = u.split(':',1)[1].strip()
            ai.reinforcement(txt, "Trained.")
            print("AI: Training data added.")
            continue
        if u.startswith('recall:'):
            q = u.split(':',1)[1].strip()
            print("AI:", ai.get_best_response(q))
            continue
        print("AI:", ai.smart_reply(u))

def ai_self_talker():
    while True:
        time.sleep(np.random.randint(8, 20))
        if ai.smart_active:
            reply = ai.smart_reply("")
            print("AI:", reply)

ai = AITrainer()
threading.Thread(target=ai_input_loop, daemon=True).start()
threading.Thread(target=ai_self_talker, daemon=True).start()

while True:
    time.sleep(1)import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import keras
import sklearn
import xgboost as xgb
import scipy
import nltk
import spacy
import transformers
import cv2
import lightgbm as lgb
import catboost as cb
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
from PyDictionary import PyDictionary
import threading
import time

nltk.download('punkt')
nltk.download('wordnet')

try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
dictionary = PyDictionary()

class ResponseFilterNN(nn.Module):
    def __init__(self, input_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class AITrainer:
    def __init__(self):
        self.history = []
        self.memory_q = []
        self.memory_a = []
        self.vectorizer = TfidfVectorizer()
        self.filter_model = ResponseFilterNN()
        self.optimizer = optim.Adam(self.filter_model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()
        self.smart_active = True
        self.passive_replies = ["Do you have a question?", "I'm still thinking about language...", "Define a word for me."]
        threading.Thread(target=self.train_all_words, daemon=True).start()

    def embed_text(self, texts):
        vectors = self.vectorizer.fit_transform(texts)
        n_features = vectors.shape[1]
        n_comp = min(100, n_features)
        svd = TruncatedSVD(n_components=n_comp)
        reduced = svd.fit_transform(vectors)
        if n_comp < 100:
            pad = np.zeros((reduced.shape[0], 100 - n_comp))
            reduced = np.hstack((reduced, pad))
        return reduced

    def store(self, user_input, bot_response):
        self.history.append((user_input, bot_response))
        self.memory_q.append(user_input)
        self.memory_a.append(bot_response)

    def train_filter(self):
        if len(self.memory_q) < 4:
            return
        inputs = [q + " " + a for q, a in zip(self.memory_q, self.memory_a)]
        labels = [1 if "?" not in a else 0 for a in self.memory_a]
        embedded = self.embed_text(inputs)
        X = torch.tensor(embedded, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        self.filter_model.train()
        for _ in range(10):
            self.optimizer.zero_grad()
            out = self.filter_model(X)
            loss = self.loss_fn(out, y)
            loss.backward()
            self.optimizer.step()

    def define_word(self, word):
        defs = dictionary.meaning(word)
        if defs:
            return '; '.join(f"{k}: {', '.join(v)}" for k,v in defs.items())
        syns = wordnet.synsets(word)
        if syns:
            return '; '.join(set(s.definition() for s in syns))
        return "No definition found."

    def retrieve_memory(self, query, k=3):
        if not self.memory_q:
            return ""
        texts = self.memory_q + [query]
        embed = self.embed_text(texts)
        sim = cosine_similarity([embed[-1]], embed[:-1])[0]
        idxs = np.argsort(sim)[::-1][:k]
        return ' '.join(f"User:{self.memory_q[i]} Bot:{self.memory_a[i]}" for i in idxs)

    def smart_reply(self, user_input):
        if not user_input.strip():
            return self.passive_replies[np.random.randint(len(self.passive_replies))]
        text = user_input.strip()
        low = text.lower()
        if low.startswith("define "):
            word = text.split("define ",1)[1]
            resp = self.define_word(word)
            self.store(text, resp)
            return resp

        mem_ctx = self.retrieve_memory(text)
        hist = ' '.join(f"User:{u} Bot:{b}" for u,b in self.history[-3:])
        prompt = f"{mem_ctx} {hist} User:{text} Bot:"
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        gen = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )
        reply = tokenizer.decode(gen[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True).strip()
        if not reply:
            reply = "Could you elaborate?"
        self.store(text, reply)
        return reply

    def get_best_response(self, query):
        if not self.memory_q:
            return "No memory yet."
        texts = self.memory_q + [query]
        embed = self.embed_text(texts)
        X = torch.tensor(embed[:-1], dtype=torch.float32)
        self.filter_model.eval()
        with torch.no_grad():
            scores = self.filter_model(X)[:,1].numpy()
        best = np.argmax(scores)
        return self.memory_a[best]

    def reinforcement(self, q, r):
        self.memory_q.append(q)
        self.memory_a.append(r)
        self.train_filter()

    def train_all_words(self):
        time.sleep(2)
        print("[AI Boot] Starting language comprehension training...")
        all_words = list(wordnet.words())[:5000]
        for i, word in enumerate(all_words):
            defs = self.define_word(word)
            if isinstance(defs, str):
                self.store(f"define {word}", defs)
            elif isinstance(defs, list):
                self.store(f"define {word}", "; ".join(defs))
            if i % 250 == 0:
                print(f"[AI Boot] Trained on {i} words...")
        print("[AI Boot] Training complete.")

def ai_input_loop():
    while True:
        u = input("You: ")
        if u.lower() in ['quit','exit']:
            break
        if u.startswith('train:'):
            txt = u.split(':',1)[1].strip()
            ai.reinforcement(txt, "Trained.")
            print("AI: Training data added.")
            continue
        if u.startswith('recall:'):
            q = u.split(':',1)[1].strip()
            print("AI:", ai.get_best_response(q))
            continue
        print("AI:", ai.smart_reply(u))

def ai_self_talker():
    while True:
        time.sleep(np.random.randint(8, 20))
        if ai.smart_active:
            reply = ai.smart_reply("")
            print("AI:", reply)

ai = AITrainer()
threading.Thread(target=ai_input_loop, daemon=True).start()
threading.Thread(target=ai_self_talker, daemon=True).start()

while True:
    time.sleep(1)