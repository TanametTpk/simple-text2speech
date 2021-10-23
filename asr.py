import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pythainlp import correct
import pathlib

fileLocation = pathlib.Path(__file__).parent.resolve()

processor = Wav2Vec2Processor.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
model = Wav2Vec2ForCTC.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
model.to("cuda")

def word_correction(sentence):
    newText = ""
    for subword in sentence.split(" "):
        if len(newText) > 0:
            newText += " " + correct(subword)
        else:
            newText = correct(subword)
    return newText

def transcribe_byte(wav_byte, sampling_rate = 16_000):
    if sampling_rate != 16_000:
        wav_byte = librosa.resample(wav_byte, sampling_rate, 16_000)

    input_values = processor(wav_byte, return_tensors = 'pt', sampling_rate=16_000).input_values

    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits
    predicted_ids = torch.argmax(logits, dim =-1)

    transcriptions = processor.decode(predicted_ids[0])
    return transcriptions