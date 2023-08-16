import json
import random
import os
import string
from Cipher import *

import random
import spacy
from spacy.lang.en import English

from Janex import *
import numpy as np

class JanexSpacy:
    def __init__(self, intents_file_path, thesaurus_file_path, vectors_file_path):
        self.JVanilla = IntentMatcher(intents_file_path, thesaurus_file_path)
        self.nlp = spacy.load("en_core_web_sm")
        self.intents = self.JVanilla.train()
        self.intentmatcher = IntentMatcher(intents_file_path, thesaurus_file_path)

        # Load or compute and save vectors
        if os.path.exists(vectors_file_path):
            with open(vectors_file_path, "r") as vectors_file:
                self.pattern_vectors = json.load(vectors_file)
        else:
            self.pattern_vectors = {}
            for intent_class in self.intents["intents"]:
                for pattern in intent_class["patterns"]:
                    pattern_vector = self.nlp(pattern).vector
                    self.pattern_vectors[pattern] = pattern_vector.tolist()  # Convert to list for JSON serialization
            with open(vectors_file_path, "w") as vectors_file:
                json.dump(self.pattern_vectors, vectors_file)

    def pattern_compare(self, input_string):
        intent_class, similarity = self.intentmatcher.pattern_compare(input_string)
        return intent_class

    def response_compare(self, input_string, intent_class):
        highest_similarity = 0
        most_similar_response = None
        threshold = 0.085

        responses = intent_class["responses"] if intent_class else []

        input_doc = self.nlp(input_string)

        for response in responses:
            if response is not None:
                response_vector = self.nlp(response).vector  # Precompute vector
                try:
                    similarity = self.calculate_cosine_similarity(input_doc.vector, response_vector)
                except:
                    return "Sorry, I did not understand your input. Try asking something else."

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar_response = response
            else:
                pass

        if most_similar_response and highest_similarity > threshold:
            return most_similar_response
        else:
            most_similar_response = self.intentmatcher.response_compare(input_string, intent_class)
            return most_similar_response

    def calculate_cosine_similarity(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        cosine_similarity = dot_product / (norm1 * norm2)
        return cosine_similarity

    def update_thesaurus(self):
        self.intentmatcher.update_thesaurus(self.thesaurus_file_path)

    def Tokenizer(self, input_string):
        words = []
        words = self.intentmatcher.tokenize(input_string)
        return words

    def ResponseGenerator(self, input_string):
        NewResponse = self.intentmatcher.ResponseGenerator(input_string)
        return NewResponse
