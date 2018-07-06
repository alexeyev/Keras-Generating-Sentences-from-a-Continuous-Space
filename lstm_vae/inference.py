# coding: utf-8
import numpy as np


def decode_sequence(states_value, decoder_adapter_model, rnn_decoder_model, num_decoder_tokens, token2id, id2token, max_seq_length):
    """
    Decoding adapted from this example:
    https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

    :param states_value:
    :param decoder_adapter_model: reads text representation, makes the first prediction, yields states after the first RNN's step
    :param rnn_decoder_model: reads previous states and makes one RNN step
    :param num_decoder_tokens:
    :param token2id: dict mapping words to ids
    :param id2token: dict mapping ids to words
    :param max_seq_length: the maximum length of the sequence
    :return:
    """

    # generate empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # populate the first token of the target sequence with the start character
    target_seq[0, 0, token2id["\t"]] = 1.0

    # sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1)
    stop_condition = False
    decoded_sentence = ""

    first_time = True
    h, c = None, None

    while not stop_condition:

        if first_time:
            # feeding in states sampled with the mean and std provided by encoder
            # and getting current LSTM states to feed in to the decoder at the next step
            output_tokens, h, c = decoder_adapter_model.predict([target_seq, states_value])
            first_time = False
        else:
            # reading output token
            output_tokens, h, c = rnn_decoder_model.predict([target_seq, h, c])

        # sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = id2token[sampled_token_index]
        decoded_sentence += sampled_token + " "

        # exit condition: either hit max length
        # or find stop character.
        if sampled_token == "<end>" or len(decoded_sentence) > max_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

    return decoded_sentence
