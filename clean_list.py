import os
from pathlib import Path
import librosa
from scipy.io import wavfile
import numpy as np
import whisper
import librosa  # Optional. Use any library you like to read audio files.
import soundfile  # Optional. Use any library you like to write audio files.

from slicer2 import Slicer

a="gxd" #此处更改人名（英文简称）

def split_long_audio(filepaths, save_dir="data_dir", out_sr=44100):
  audio, sr = librosa.load(f'{filepaths}.wav', sr=None, mono=False)  # Load an audio file with librosa.
  slicer = Slicer(
      sr=sr,
      threshold=-40,
      min_length=2000,
      min_interval=300,
      hop_size=10,
      max_sil_kept=500
  )
  chunks = slicer.slice(audio)
  for i, chunk in enumerate(chunks):
      if len(chunk.shape) > 1:
          chunk = chunk.T  # Swap axes if the audio is stereo.
      soundfile.write(f'{save_dir}/{a}_{i}.wav', chunk, sr)  # Save sliced audio files with soundfile.

  if os.path.exists(f'{filepaths}'):  # 如果文件存在
      os.remove(f'{filepaths}')  

def transcribe_one(audio_path): # 使用whisper语音识别
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)
    return result.text 

if __name__ == '__main__':
    whisper_size = "medium"
    model = whisper.load_model(whisper_size)
    audio_path = f"./raw/{a}"
    
    if os.path.exists(audio_path):
	    for filename in os.listdir(audio_path): # 删除原来的音频和文本
	        file_path = os.path.join(audio_path, filename)
	        os.remove(file_path)
    split_long_audio(f"data/{a}", f"./raw/{a}")
    files=os.listdir(audio_path)
    file_list_sorted = sorted(files, key=lambda x: int(os.path.splitext(x)[0].split('_')[1]))
    filepaths=[os.path.join(audio_path,i)  for i in file_list_sorted]
    for file_idx, filepath in enumerate(filepaths):  # 循环使用whisper遍历每一个音频,写入.alb
        text = transcribe_one(filepath)
        with open(f"./raw/{a}/{a}_{file_idx}.lab",'w') as f:
            f.write(text)



