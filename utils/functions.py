import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import librosa


def read_wav_as_np_array(audio_bytes):
    # Save the audio bytes to a temporary file
    with open("temp.wav", "wb") as file:
        file.write(audio_bytes)

    # Read the temporary WAV file using scipy's wavfile module
    audio_data, sample_rate = librosa.load("temp.wav", sr=None)
    # Convert the audio data to a NumPy array
    audio_array = torch.tensor(audio_data, dtype=torch.float32)

    # Remove the temporary WAV file
    os.remove("temp.wav")

    return sample_rate, audio_array

def process_input(model, data_processing, string, vocab, device):
    encoded_sequence = data_processing.process([string])
    prediction = model.greedy_inference(src=encoded_sequence,
                                             sos_idx=2,
                                             eos_idx=3,
                                             max_length=50,
                                             device=device)
    print(prediction)
    itos = vocab.itos.copy()
    str_list = ["".join([itos[idx] for idx in row[1:-1]]) for row in prediction.tolist()]
    return str_list[0]

def attention_heatmap(model):
    encoder_layers = model.encoder.layers
    decoder_layers = model.decoder.layers

    encoder_attention_weights = []
    for layer in encoder_layers:
        attention_weights = layer.MHA.attention_weights
        encoder_attention_weights.append(attention_weights)

    decoder_self_attention_weights = []
    decoder_enc_attention_weights = []
    for layer in decoder_layers:
        self_attention_weights = layer.selfAttn.attention_weights
        enc_attention_weights = layer.encAttn.attention_weights
        decoder_self_attention_weights.append(self_attention_weights)
        decoder_enc_attention_weights.append(enc_attention_weights)
    
    return encoder_attention_weights, decoder_self_attention_weights, decoder_enc_attention_weights

def visualize_token2token_scores(scores_mat, chars, x_label_name='Head', plots_per_row=4):
    num_heads = len(scores_mat)
    num_rows = num_heads // plots_per_row
    if num_heads % plots_per_row != 0:
        num_rows += 1

    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(plots_per_row*4, num_rows*4))

    if num_rows == 1:
        axes = np.expand_dims(axes, 0)

    for idx, scores in enumerate(scores_mat):
        row = idx // plots_per_row
        col = idx % plots_per_row
        ax = axes[row, col]

        im = ax.imshow(scores, cmap='viridis')

        ax.set_xticks(range(len(chars)))
        ax.set_yticks(range(len(chars)))

        ax.set_xticklabels(chars, rotation=90)
        ax.set_yticklabels(chars)

        ax.set_xlabel('{} {}'.format(x_label_name, idx+1))

        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()

    return fig

def visualize_token2token_scores_enc_dec(scores_mat, in_chars, out_chars, x_label_name='Head', plots_per_row=4):
    num_heads = len(scores_mat)
    num_rows = num_heads // plots_per_row
    if num_heads % plots_per_row != 0:
        num_rows += 1

    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(plots_per_row*4, num_rows*4))

    if num_rows == 1:
        axes = np.expand_dims(axes, 0)

    for idx, scores in enumerate(scores_mat):
        row = idx // plots_per_row
        col = idx % plots_per_row
        ax = axes[row, col]

        im = ax.imshow(scores, cmap='viridis')

        ax.set_xticks(range(len(in_chars)))
        ax.set_yticks(range(len(out_chars)))

        ax.set_xticklabels(in_chars, rotation=90)
        ax.set_yticklabels(out_chars)

        ax.set_xlabel('{} {}'.format(x_label_name, idx+1))

        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()

    return fig