import streamlit as st
import importlib
import torch
from torchtext.data import Field
from easydict import EasyDict as edict
from utils.functions import process_input, attention_heatmap, visualize_token2token_scores, visualize_token2token_scores_enc_dec, read_wav_as_np_array
from utils.st_custom_components import st_audiorec
from utils.parser import MathParser
from models.asr_model import ASRModel
import os

VOCAB_PATH = os.path.join("assets", "vocab.pt")
AVAILABLE_ASR_MODELS = {
    "Wav2Vec2": {
        "model_name": "wav2vec2"
    },
    "Whisper (large)": {
        "model_name": "whisper-large-v2"
    },
    "Whisper (medium)": {
        "model_name": "whisper-medium"
    }
}
AVAILABLE_MODELS = {
    "small (290 MB)": {
        "model_path": r"C:\Users\anteg\Desktop\PROJEKTI\TP-Transformer-master\trained\lr=0.0001_bs=128_h=512_f=2048_nl=3_nh=4_d=0.0_Adam_\195932581\best_eval_model.pt",
        "parameters": edict({'model_name':None, 'log_folder':'log', 'dropout':0.0, 'hidden':512, 'n_layers':3, 'n_heads':4, 'filter':2048, 'd_r':0, 'input_dim':73})
    },
    "medium (577 MB)": {
        "model_path": r"C:\Users\anteg\Desktop\PROJEKTI\TP-Transformer-master\trained\lr=0.0001_bs=128_h=512_f=2048_nl=6_nh=8_d=0.0_Adam_\195932581\best_eval_model.pt",
        "parameters": edict({'model_name':None, 'log_folder':'log', 'dropout':0.0, 'hidden':512, 'n_layers':6, 'n_heads':8, 'filter':2048, 'd_r':0, 'input_dim':73})
    },
    "large (766 MB)": {
        "model_path": r"C:\Users\anteg\Desktop\PROJEKTI\TP-Transformer-master\trained\lr=0.0001_bs=128_h=512_f=2048_nl=8_nh=3_d=0.0_Adam_\195932581\best_eval_model.pt",
        "parameters": edict({'model_name':None, 'log_folder':'log', 'dropout':0.0, 'hidden':512, 'n_layers':8, 'n_heads':3, 'filter':2048, 'd_r':0, 'input_dim':73})
    }
}
math_parser = MathParser()
vocab = torch.load(VOCAB_PATH)
imp_module = importlib.import_module("models.tp-transformer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
split_chars = lambda x: list(x)
data_processing = Field(tokenize=split_chars,
                init_token='<sos>',
                eos_token='<eos>',
                batch_first=True)
data_processing.vocab = vocab

@st.cache_resource
def load_asr_model(model_name):
    return ASRModel(model_name)

@st.cache_resource
def load_math_model(model_path, parameters):
    model = imp_module.build_transformer(parameters, 1)
    model.load_state_dict(torch.load(model_path)["model"])
    model = model.to(device)
    return model


st.set_page_config(page_title='Vox Calculi', page_icon='🧮', initial_sidebar_state='auto')

st.sidebar.header("Please select a transformer model:")  
model_selector = st.sidebar.selectbox("Model", list(AVAILABLE_MODELS.keys()))
st.sidebar.divider()
st.sidebar.header("Please select ASR model:")
asr_selector = st.sidebar.selectbox("ASR model", list(AVAILABLE_ASR_MODELS.keys()))

model_path = AVAILABLE_MODELS[model_selector]["model_path"]
parameters = AVAILABLE_MODELS[model_selector]["parameters"]
model = load_math_model(model_path, parameters)
asr_model = load_asr_model(AVAILABLE_ASR_MODELS[asr_selector]["model_name"])

st.title("🧮 Vox Calculi")
st.markdown("### Please speak your math problem:")
wav_audio_data = st_audiorec()
uploaded_file = st.file_uploader("Or upload a .wav file", type=['wav'])
if uploaded_file is not None:
    wav_audio_data = uploaded_file.read()

if 'prev_audio_data' not in st.session_state:
    st.session_state.prev_audio_data = None
if 'sequence' not in st.session_state:
    st.session_state.sequence = ""

if wav_audio_data is not None and (st.session_state.prev_audio_data is None or st.session_state.prev_audio_data != wav_audio_data):
    sample_rate, audio_data = read_wav_as_np_array(wav_audio_data)
    if audio_data.size() != torch.Size([0]):
        st.session_state.prev_audio_data = wav_audio_data
        transcription = asr_model(audio_data, sample_rate)
        print(transcription)
        st.session_state.sequence = math_parser(transcription)

if st.session_state.prev_audio_data is not None and st.session_state.sequence.strip() != "":
    st.divider()
    st.markdown("### Transcribed voice input")
    sequence_input = st.text_input('', st.session_state.sequence)
else:
    sequence_input = ""

if st.session_state.prev_audio_data is not None and len(st.session_state.prev_audio_data) > 0:
    if st.button('Submit'):
        st.divider()
        sequence = sequence_input
        st.markdown("##### Answer:")
        answer = process_input(model, data_processing, sequence, vocab, device)
        st.write(answer)
        chars = split_chars(sequence)
        chars.insert(0, "<sos>")
        chars.append("<eos>")
        answer_chars = split_chars(answer)
        answer_chars.insert(0, "<sos>")

        encoder_attention_weights, decoder_self_attention_weights, decoder_enc_attention_weights = attention_heatmap(model)

        encoder_expander = st.expander("Encoder Attention Heatmaps")
        encoder_tabs = encoder_expander.tabs(["Encoder " + str(i+1) for i in range(len(encoder_attention_weights))])
        for i in encoder_attention_weights:
            print(i.shape)
        for idx, tab in enumerate(encoder_tabs):
            with tab:
                fig = visualize_token2token_scores(encoder_attention_weights[idx].squeeze().detach().cpu().numpy(), chars, plots_per_row=4)
                st.pyplot(fig, use_container_width=True)
        decoder_expander = st.expander("Decoder Attention Heatmaps")
        decoder_tabs = decoder_expander.tabs(["Decoder " + str(i+1) for i in range(len(decoder_self_attention_weights))])
        for i in decoder_self_attention_weights:
            print(i.shape)
        for idx, tab in enumerate(decoder_tabs):
            with tab:
                fig = visualize_token2token_scores(decoder_self_attention_weights[idx].squeeze().detach().cpu().numpy(), answer_chars, plots_per_row=4)
                st.pyplot(fig, use_container_width=True)
        decoder_enc_expander = st.expander("Decoder-Encoder Attention Heatmaps")
        decoder_enc_tabs = decoder_enc_expander.tabs(["Decoder-Encoder " + str(i+1) for i in range(len(decoder_enc_attention_weights))])
        for i in decoder_enc_attention_weights:
            print(i.shape)
        for idx, tab in enumerate(decoder_enc_tabs):
            with tab:
                fig = visualize_token2token_scores_enc_dec(decoder_enc_attention_weights[idx].squeeze().detach().cpu().numpy(), chars, answer_chars, plots_per_row=4)
                st.pyplot(fig, use_container_width=True)
else:
    st.button('Submit', disabled=True)