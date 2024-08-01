# decoder.py

import numpy as np

def greedy_decoder(predictions):
    """Greedy decoder for converting model output to text."""
    decoded_output = []
    for prediction in predictions:
        decoded_output.append(np.argmax(prediction))
    return decoded_output

def beam_search_decoder(predictions, beam_width=3):
    """Beam search decoder for converting model output to text."""
    # Placeholder implementation for beam search
    decoded_output = []
    for prediction in predictions:
        beam = [(-np.inf, [])] * beam_width
        for token in prediction:
            new_beam = []
            for prob, path in beam:
                for char in range(len(token)):
                    new_path = path + [char]
                    new_prob = prob + np.log(token[char])
                    new_beam.append((new_prob, new_path))
            beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]
        decoded_output.append(beam[0][1])
    return decoded_output
