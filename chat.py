"""Command-line intent-classification demo; importing this module does not start input."""
import json
import random
from pathlib import Path
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

ROOT = Path(__file__).resolve().parent
FALLBACK = "I'm not sure I understand. Can you try rephrasing?"


class MoodMate:
    def __init__(self, checkpoint=ROOT / 'data.pth', intents_path=ROOT / 'intents.json', threshold=0.75):
        self.threshold = threshold
        with open(intents_path, encoding='utf-8') as handle:
            self.intents = json.load(handle)['intents']
        data = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.words = data['all_words']
        self.tags = data['tags']
        self.model = NeuralNet(data['input_size'], data['hidden_size'], data['output_size'])
        self.model.load_state_dict(data['model_state'])
        self.model.eval()
        self.responses = {intent['tag']: intent['responses'] for intent in self.intents}

    def reply(self, sentence):
        if not sentence.strip():
            return FALLBACK
        features = bag_of_words(tokenize(sentence), self.words)
        # A network bias can be confident on an entirely unknown input.
        if not features.any():
            return FALLBACK
        tensor = torch.from_numpy(features).float().unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)
            probability, index = probabilities.max(dim=1)
        if probability.item() <= self.threshold:
            return FALLBACK
        options = self.responses.get(self.tags[index.item()], [])
        return random.choice(options) if options else FALLBACK


def main():
    bot = MoodMate()
    print('MoodMate: educational intent-classification demo, not clinical advice.')
    print("Type 'quit' to stop.")
    while True:
        try:
            sentence = input('You: ')
        except (EOFError, KeyboardInterrupt):
            print(); break
        if sentence.strip().lower() == 'quit':
            break
        print('MoodMate:', bot.reply(sentence))


if __name__ == '__main__':
    main()
