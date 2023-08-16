from main import *

intents_file_path = "./intents.json"
thesaurus_file_path = "./thesaurus.json"
vectors_file_path = "./vectors.json"

matcher = JanexSpacy(intents_file_path, thesaurus_file_path, vectors_file_path)

input_string = input("You: ")

#matcher.train_model()

intent_class = matcher.pattern_compare(input_string)
response = matcher.response_compare(input_string, intent_class)

print(response)
