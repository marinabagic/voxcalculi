from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from torchaudio.transforms import Resample
device = torch.device('cuda:0') if torch.cuda.is_available() else "cpu"

class ASRModel:
    def __init__(self, model_name="wav2vec2"):
        if model_name == "wav2vec2":
            self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
            self.model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        if model_name == "whisper-large-v2":
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
            self.model.config.forced_decoder_ids = None
        if model_name == "whisper-medium":
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
            self.model.config.forced_decoder_ids = None
            self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-medium",
            chunk_length_s=30,
            device=device,
            )

        self.model_name = model_name

    def preprocess(self, audio_data, sample_rate):
        if sample_rate != 16000:
            resample = Resample(sample_rate, 16000)
            audio_data = resample(audio_data)
        return audio_data

    def process_audio(self, audio_data, sample_rate):
        audio = self.preprocess(audio_data, sample_rate)
        print(type(audio))
        if self.model_name.startswith("whisper"):
            prediction = self.pipe(audio.numpy())["text"]
            print(prediction)
            return prediction
        if self.model_name == "wav2vec2":
            inputs = self.processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits


            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)
            return transcription[0]
    
    def __call__(self, audio_data, sample_rate):
        return self.process_audio(audio_data, sample_rate)