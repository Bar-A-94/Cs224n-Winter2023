# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils

correct = 0
total = 0
with open('/Users/bara/PycharmProjects/cs224n/a5/code/london.dev.predictions', 'w', encoding='utf-8') as fout:
    predictions = ['London'] * len(open('/Users/bara/PycharmProjects/cs224n/a5/code/birth_dev.tsv', "r").readlines())
    total, correct = utils.evaluate_places('/Users/bara/PycharmProjects/cs224n/a5/code/birth_dev.tsv', predictions)
if total > 0:
    print('Correct: {} out of {}: {}%'.format(correct, total, correct / total * 100))
else:
    print('Predictions written to {}; no targets provided'
          .format('/Users/bara/PycharmProjects/cs224n/a5/code/london.dev.predictions'))

# Correct: 25.0 out of 500.0: 5.0%
