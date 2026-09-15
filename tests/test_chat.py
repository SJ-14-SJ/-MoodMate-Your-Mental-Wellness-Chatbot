import unittest
from unittest.mock import patch
import numpy as np
from chat import MoodMate, FALLBACK
from nltk_utils import bag_of_words, tokenize

class ChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = MoodMate()

    def test_checkpoint_loads_on_cpu(self):
        self.assertEqual(next(self.bot.model.parameters()).device.type, 'cpu')

    def test_empty_input(self):
        self.assertEqual(self.bot.reply('  '), FALLBACK)

    def test_unknown_words_use_fallback(self):
        self.assertEqual(self.bot.reply('zxqvunknownword123'), FALLBACK)

    def test_bag_of_words_uses_stems(self):
        np.testing.assert_array_equal(bag_of_words(tokenize('Running running'), ['run', 'walk']), [1, 0])

    def test_low_confidence_fallback(self):
        # Threshold above any probability makes this independent of checkpoint accuracy.
        with patch.object(self.bot, 'threshold', 1.0):
            self.assertEqual(self.bot.reply('hello'), FALLBACK)

    def test_reply_is_template_or_fallback(self):
        valid = {FALLBACK}
        for responses in self.bot.responses.values():
            valid.update(responses)
        self.assertIn(self.bot.reply('hello'), valid)

if __name__ == '__main__':
    unittest.main()
