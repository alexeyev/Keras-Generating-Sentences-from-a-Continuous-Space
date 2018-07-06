# coding: utf-8

import numpy as np
from nltk.tokenize import word_tokenize
from lstm_vae import create_lstm_vae, inference


def get_text_data(data_path, num_samples=1000):

    # vectorize the data
    input_texts = []
    input_characters = set(["\t"])

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.read().lower().split("\n")

    for line in lines[: min(num_samples, len(lines) - 1)]:

        input_text, _ = line.split("\t")
        input_text = word_tokenize(input_text)
        input_text.append("<end>")

        input_texts.append(input_text)

        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)

    input_characters = sorted(list(input_characters))
    num_encoder_tokens = len(input_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts]) + 1

    print("Number of samples:", len(input_texts))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    decoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")

    for i, input_text in enumerate(input_texts):
        decoder_input_data[i, 0, input_token_index["\t"]] = 1.0

        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
            decoder_input_data[i, t + 1, input_token_index[char]] = 1.0

    return max_encoder_seq_length, num_encoder_tokens, input_characters, input_token_index, reverse_input_char_index, \
           encoder_input_data, decoder_input_data


if __name__ == "__main__":

    timesteps_max, enc_tokens, characters, char2id, id2char, x, x_decoder = get_text_data(num_samples=3000,
                                                                                          data_path="data/fra.txt")

    print(x.shape, "Creating model...")

    input_dim = x.shape[-1]
    timesteps = x.shape[-2]
    batch_size = 1
    latent_dim = 191
    intermediate_dim = 353
    epochs = 40

    vae, enc, gen, stepper = create_lstm_vae(input_dim,
                                             batch_size=batch_size,
                                             intermediate_dim=intermediate_dim,
                                             latent_dim=latent_dim)
    print("Training model...")

    vae.fit([x, x_decoder], x, epochs=epochs, verbose=1)

    print("Fitted, predicting...")


    def decode(s):
        return inference.decode_sequence(s, gen, stepper, input_dim, char2id, id2char, timesteps_max)


    for _ in range(5):

        id_from = np.random.randint(0, x.shape[0] - 1)
        id_to = np.random.randint(0, x.shape[0] - 1)

        m_from, std_from = enc.predict([[x[id_from]]])
        m_to, std_to = enc.predict([[x[id_to]]])

        seq_from = np.random.normal(size=(latent_dim,))
        seq_from = m_from + std_from * seq_from

        seq_to = np.random.normal(size=(latent_dim,))
        seq_to = m_to + std_to * seq_to

        print("==  \t", " ".join([id2char[j] for j in np.argmax(x[id_from], axis=1)]), "==")

        for v in np.linspace(0, 1, 7):
            print("%.2f\t" % (1 - v), decode(v * seq_to + (1 - v) * seq_from))

        print("==  \t", " ".join([id2char[j] for j in np.argmax(x[id_to], axis=1)]), "==")