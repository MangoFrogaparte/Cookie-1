import os
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Module, Linear, ReLU, Dropout, Softmax



class Cookie(torch.nn.Module):

  def __init__(self, input_size, output_size):
    super(Cookie, self).__init__()

    self.fcl = nn.Linear(input_size, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, output_size)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    x = self.relu(self.fcl(x))
    x = self.dropout(x)
    x = self.relu(self.fc2(x))
    x = self.dropout(x)
    x = self.fc3(x)
    return x

class ChatbotAssistant:
  def __init__(self, intents_path, function_mappings=None):
    self.model = None
    self.intents_path = intents_path
    self.documents = []
    self.vocabulary = []
    self.labels = []
    self.intents = []
    self.intents_responses = {}
    self.function_mappings = function_mappings
    self.X = None
    self.y = None
    self.lemmatizer = nltk.WordNetLemmatizer()

  @staticmethod
  def tokenize_and_lemmatize(text):
    lemmatizer = nltk.WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words

  def bag_of_words(self, words):
    bag = [0] * len(self.vocabulary)
    for w in words:
        if w in self.vocabulary:
            bag[self.vocabulary.index(w)] = 1
    return bag

  def parse_intents(self):
    with open(self.intents_path, 'r') as f:
      intents_data = json.load(f)

    for intent in intents_data['intents']:
        tag = intent['tag']
        if tag not in self.intents:
            self.intents.append(tag)
        self.intents_responses[tag] = intent['responses']

        for pattern in intent['patterns']:
            pattern_words = ChatbotAssistant.tokenize_and_lemmatize(pattern)
            self.vocabulary.extend(pattern_words)
            self.documents.append((pattern_words, tag))

    self.vocabulary = sorted(list(set(self.vocabulary)))

  def prepare_data(self):
    bags = []
    indices = []

    for document in self.documents:
      words = document[0]
      bag = self.bag_of_words(words)

      intent_tag = document[1]
      if intent_tag in self.intents:
          intent_index = self.intents.index(intent_tag)
          bags.append(bag)
          indices.append(intent_index)
      else:
          print(f"Warning: Intent tag '{intent_tag}' not found in self.intents. Skipping document.")


    self.X = np.array(bags)
    self.y = np.array(indices)

  def train_model(self, batch_size, lr, epochs):
    X_tensor = torch.tensor(self.X, dtype=torch.float32)
    y_tensor = torch.tensor(self.y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    self.model = Cookie(self.X.shape[1], len(self.intents))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(self.model.parameters(), lr=lr)

    for epoch in range(epochs):
      running_loss = 0.0

      for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = self.model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

      print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")

  def save_model(self, model_path, dimensions_path):
    torch.save(self.model.state_dict(), model_path)

    with open(dimensions_path, 'w') as f:
      json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents) }, f)

  def load_model(self, model_path, dimensions_path):
    with open(dimensions_path, 'r') as f:
      dimensions = json.load(f)

    self.model = Cookie(dimensions['input_size'], dimensions['output_size'])
    self.model.load_state_dict(torch.load(model_path))
    self.model.eval()

  def process_message(self, input_message):
    words = ChatbotAssistant.tokenize_and_lemmatize(input_message)
    bag = self.bag_of_words(words)

    bag_tensor = torch.tensor([bag], dtype=torch.float32)
    self.model.eval()
    with torch.no_grad():
      predictions = self.model(bag_tensor)

    predicted_class_index = torch.argmax(predictions, dim=1).item()
    predicted_intent = self.intents[predicted_class_index]

    if self.function_mappings:
        if predicted_intent in self.function_mappings:
            self.function_mappings[predicted_intent]()

    if predicted_intent in self.intents_responses and self.intents_responses[predicted_intent]:
        return random.choice(self.intents_responses[predicted_intent])
    else:
        return f"Sorry, I don't have a response for '{predicted_intent}'."


if __name__ == '__main__':
  if not os.path.exists('intents.json'):
      dummy_intents = {
          "intents": [
              {"tag": "greeting",
               "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day"],
               "responses": ["Hello!", "Good to see you again!", "Hi there, how can I help?"]
              },
              {"tag": "goodbye",
               "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you", "Till next time"],
               "responses": ["See you!", "Have a nice day", "Bye! Come back again soon."]
              },
              {"tag": "name",
               "patterns": ["What is your name?", "Who are you?", "May I know your name?"],
               "responses": ["I am a chatbot.", "You can call me Cookie.", "I don't have a name."]
              }
          ]
      }
      with open('intents.json', 'w') as f:
          json.dump(dummy_intents, f, indent=4)

  assistant = ChatbotAssistant('intents.json')
  assistant.parse_intents()
  assistant.prepare_data()
  assistant.train_model(batch_size=8, lr=0.001, epochs=200)
  assistant.save_model('Cookie.pth', 'dimensions.json')
  assistant.load_model('Cookie.pth', 'dimensions.json')

  print("Cookie is ready! Type 'exit' to quit.")
  while True:
    message = input('You: ')
    if message.lower() in ['bye', 'later', 'quit', 'goodbye', 'exit']:
      print("Cookie: Goodbye!")
      break

    response = assistant.process_message(message)
    print(f"Cookie: {response}")