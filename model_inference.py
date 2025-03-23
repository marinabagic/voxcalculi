import torch
import importlib
from easydict import EasyDict as edict
from torchtext.data import Field
from matplotlib import pyplot as plt
import numpy as np

MODEL_PATH = r"C:\Users\anteg\Desktop\PROJEKTI\TP-Transformer-master\trained\tp-transformer-saved-attention.pt"
VOCAB_PATH = r"C:\Users\anteg\Desktop\PROJEKTI\TP-Transformer-master\vocab.pt"

def process_input(model, data_processing, string, vocab, device):
    encoded_sequence = data_processing.process([string])
    prediction = model.greedy_inference(src=encoded_sequence,
                                             sos_idx=2,
                                             eos_idx=3,
                                             max_length=50,
                                             device=device)
    
    itos = vocab.itos.copy()
    str_list = ["".join([itos[idx] for idx in row[1:-1]]) for row in prediction.tolist()]
    return str_list[0]

def visualize_token2token_scores(scores_mat, chars, x_label_name='Head'):
    num_heads = len(scores_mat)
    num_tokens = len(chars)
    
    num_rows = num_heads // 4 
    if num_heads % 4 != 0:
        num_rows += 1

    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5*num_rows))

    for idx, scores in enumerate(scores_mat):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        im = ax.imshow(scores, cmap='viridis')

        ax.set_xticks(range(num_tokens))
        ax.set_yticks(range(num_tokens))

        ax.set_xticklabels(chars, rotation=90)
        ax.set_yticklabels(chars)

        ax.set_xlabel('{} {}'.format(x_label_name, idx+1))

        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()

    plt.show()

def visualize_token2head_scores(scores_mat):
    fig = plt.figure(figsize=(30, 50))

    for idx, scores in enumerate(scores_mat):
        scores_np = np.array(scores)
        ax = fig.add_subplot(6, 2, idx+1)
        # append the attention weights
        im = ax.matshow(scores_np, cmap='viridis')

        fontdict = {'fontsize': 20}

        ax.set_xticks(range(len(all_tokens)))
        ax.set_yticks(range(len(scores)))

        ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(range(len(scores[0])), fontdict=fontdict)
        ax.set_xlabel('Layer {}'.format(idx+1))

        fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def attention_heatmap(model):
    encoder_layers = model.encoder.layers
    decoder_layers = model.decoder.layers

    # Access attention weights in the encoder layers
    encoder_attention_weights = []
    for layer in encoder_layers:
        attention_weights = layer.MHA.attention_weights
        encoder_attention_weights.append(attention_weights)

    # Access attention weights in the decoder layers
    decoder_self_attention_weights = []
    decoder_enc_attention_weights = []
    for layer in decoder_layers:
        self_attention_weights = layer.selfAttn.attention_weights
        enc_attention_weights = layer.encAttn.attention_weights
        decoder_self_attention_weights.append(self_attention_weights)
        decoder_enc_attention_weights.append(enc_attention_weights)
    
    return encoder_attention_weights, decoder_self_attention_weights, decoder_enc_attention_weights

imp_module = importlib.import_module("models.tp-transformer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p = edict({'model_name':None, 'log_folder':'log', 'dropout':0.0, 'hidden':512, 'n_layers':6, 'n_heads':8, 'filter':2048, 'd_r':0, 'input_dim':73})

vocab = torch.load(VOCAB_PATH)
model = imp_module.build_transformer(p, 1)
model.load_state_dict(torch.load(MODEL_PATH)["model"])
model = model.to(device)
split_chars = lambda x: list(x)

data_processing = Field(tokenize=split_chars,
                   init_token='<sos>',
                   eos_token='<eos>',
                   batch_first=True)
data_processing.vocab = vocab

while(True):
    QUESTION = input("Enter a prompt: ")
    answer = process_input(model, data_processing, QUESTION, vocab, device)
    print("Answer: ", answer)

    encoder_attention_weights, decoder_self_attention_weights, decoder_enc_attention_weights = attention_heatmap(model)
    chars = split_chars(QUESTION)
    chars.insert(0, "<sos>")
    chars.append("<eos>")
    visualize_token2token_scores(encoder_attention_weights[-1].squeeze().detach().cpu().numpy(), chars)