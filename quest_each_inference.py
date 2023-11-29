# flake8: noqa: E402

import sys, os
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

import torch
import argparse
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
import gradio as gr
import webbrowser
import numpy as np
from scipy.io import wavfile

from pathlib import Path
import glob
import time
import shutil
import wave

import requests
from pydub import AudioSegment

net_g = None

if sys.platform == "darwin" and torch.backends.mps.is_available():
    device = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    device = "cuda"


def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str, device)
    del word2ph
    assert bert.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JP":
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    else:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language


def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language):
    global net_g
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps)
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers
        torch.cuda.empty_cache()
        return audio


def tts_fn(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language):
    slices = text.split("|")
    audio_list = []
    with torch.no_grad():
        for slice in slices:
            audio = infer(slice, sdp_ratio=sdp_ratio, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale, sid=speaker, language=language)
            audio_list.append(audio)
            silence = np.zeros(hps.data.sampling_rate)  # 生成1秒的静音
            audio_list.append(silence)  # 将静音添加到列表中
    audio_concat = np.concatenate(audio_list)
    return "Success", (hps.data.sampling_rate, audio_concat)
	
def tts_fn_create(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language):
    slices = text.split("|")
    audio_list = []
    with torch.no_grad():
        for slice in slices:
            audio = infer(slice, sdp_ratio=sdp_ratio, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale, sid=speaker, language=language)
            audio_list.append(audio)
            # silence = np.zeros(hps.data.sampling_rate)  # 生成1秒的静音
            # audio_list.append(silence)  # 将静音添加到列表中
    audio_concat = np.concatenate(audio_list)
    return audio_concat



def inference_wav(output_path,text,output_name,args,hps):
    generated_audio = tts_fn_create(text, args.speaker, args.sdp_ratio, args.noise_scale, args.noise_scale_w,
                                    args.length_scale, args.language)

    # 将音频数据转换为16位有符号整数格式
    generated_audio = (generated_audio * (2 ** 15 - 1)).astype(np.int16)

    # 保存为WAV文件
    # 保存生成的音频为.wav文件
    output_path = output_path + "/" + output_name + ".wav"  # 修改保存路径
    wavfile.write(output_path, hps.data.sampling_rate, generated_audio)

    print(f"音频已保存到{output_path}")


#把directory目录下的wav文件写入列表，并把列表从小到大排序：
def sort_files_by_number(directory):
    # 读取目录中的所有.wav文件
    audio_files = glob.glob(os.path.join(directory, '*.wav'))

    # 获取文件名列表并去重
    file_names = list(set([os.path.basename(file) for file in audio_files]))

    # 按照数字大小对文件名列表进行排序
    sorted_file_names = sorted(file_names, key=lambda name: int(name.split('.')[0]))

    return sorted_file_names

# 自己添加方法：通过给定目录，合成新的wav文件到该目录下,is_delete是否删除子文件
def merge_from_list(directory, is_delete):
    wav_list = sort_files_by_number(directory)
    sounds = []
    for wav in wav_list:
        sounds.append(AudioSegment.from_wav(directory / wav))
    playlist = AudioSegment.empty()
    for sound in sounds:
        playlist += sound
    # 导出音频并获取导出的音频文件对象
    exported_audio = playlist.export(directory / "result.wav", format="wav")
    # 释放导出的音频文件资源
    exported_audio.close()  # 关闭音频文件对象
    # 等待一段时间
    # time.sleep(2)  # 例如，等待 2 秒
    if is_delete:
        for wav in wav_list:
            os.remove(directory / wav)

# 自己添加方法：生成音频和srt文件
def create_audio(args,hps):
    while True:
        results = requests.post('https://www.yourdomain.com/api/human/findten', data={"token": "yourtoken"})
        if results.status_code == 200:
            if results.json()['data']:
                for data in results.json()['data']:
                    text = data['copywriting']
                    id = data['id']
                    wav_output_dir = "inference_demo"

                    # 处理开始
                    # 自己重新建立一个新文件夹
                    # temp_output_dir = Path(wav_output_dir,str(int(find_max_directory(wav_output_dir))+1))
                    temp_output_dir = Path(wav_output_dir, str(id))
                    if os.path.exists(temp_output_dir):
                        shutil.rmtree(temp_output_dir)
                    text_list = text.split('\n')
                    text_list = [element for element in text_list if element != ""]
                    output_dir = temp_output_dir
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # 列表形式
                    # 同名文件夹下创建z.srt
                    f2 = open(output_dir / "z.srt", 'w', encoding='utf-8-sig')

                    start_time = 0
                    end_time = 0



                    for utt_id, sentence in enumerate(text_list):
                        str_utt_id = str(utt_id + 1)

                        # 保存地址
                        wav_file = output_dir / (str_utt_id + ".wav")

                        # 生成音频保存文件
                        inference_wav(str(output_dir), sentence, str_utt_id, args, hps)

                        # 创建一个持续时间为0.3秒的空白音频
                        silence_wave = AudioSegment.silent(duration=0)  # 0.05秒的留白，单位是毫秒

                        # 加载音频文件
                        audio = AudioSegment.from_file(str(wav_file), format="wav")
                        # 放慢音频速度
                        #audio = audio.set_frame_rate(int(audio.frame_rate * 0.5))  # 播放速度减慢1半

                        slower_audio_final = audio + silence_wave
                        # 将处理后的音频保存为 WAV 文件
                        slower_audio_final.export(wav_file, format="wav")

                        # 读取文件时长
                        with wave.open(str(wav_file), 'rb') as fwave:
                            time_count = fwave.getparams().nframes / fwave.getparams().framerate
                        time_count = format(time_count, '.3f')

                        end_time = start_time + float(time_count)

                        # 计算小时、分钟和秒
                        start_hours = int(start_time // 3600)
                        start_minutes = int((start_time % 3600) // 60)
                        start_seconds = start_time % 60
                        start_milliseconds = int((start_time % 1) * 1000)

                        # 将小时、分钟、秒和毫秒格式化为两位数
                        start_time_str = "{:02d}:{:02d}:{:02d},{:03d}".format(start_hours, start_minutes,
                                                                              int(start_seconds), start_milliseconds)

                        # 计算小时、分钟和秒
                        end_hours = int(end_time // 3600)
                        end_minutes = int((end_time % 3600) // 60)
                        end_seconds = end_time % 60
                        end_milliseconds = int((end_time % 1) * 1000)

                        # 将小时、分钟、秒和毫秒格式化为两位数
                        end_time_str = "{:02d}:{:02d}:{:02d},{:03d}".format(end_hours, end_minutes, int(end_seconds),
                                                                            end_milliseconds)

                        f2.write(str_utt_id + '\n')
                        f2.write(str(start_time_str) + " --> " + str(end_time_str) + '\n')
                        f2.write(sentence + '\n\n')

                        start_time = end_time

                    # 合并音频
                    merge_from_list(output_dir, True)
                    f2.close()

                    # 处理结束

                    # 修改状态
                    requests.post('https://www.yourdomain.com/api/human/changestatus',
                                  data={"token": "yourtoken", "status": id, "id": data['id']})
            else:
                print('没有数据')
                break
        else:
            print('请求错误')
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="./logs/gxd/G_6000.pth", help="path of your model"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="./configs/config.json",
        help="path of your config file",
    )
    parser.add_argument(
        "--share", default=False, help="make link public", action="store_true"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable DEBUG-LEVEL log"
    )
    parser.add_argument(
        "-t", "--text", default="吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。", type=str, help="text to gernate voice"
    )
    parser.add_argument(
        "-s", "--speaker", default="gxd", type=str, help="speaker short name"
    )
    parser.add_argument(
        "-r", "--sdp_ratio", default=0.2, action="store_true", help="sdp ratio"
    )
    parser.add_argument(
        "--noise_scale", default=0.2, action="store_true", help="noise scale" #minimum=0.1, maximum=2
    )
    parser.add_argument(
        "--noise_scale_w", default=0.4, action="store_true", help="noise scale w" #minimum=0.1, maximum=2
    )
    parser.add_argument(
        "--length_scale", default=0.8, action="store_true", help="length scale" #minimum=0.1, maximum=2
    )
    parser.add_argument(
        "--language", default="ZH", action="store_true", help="language" #["ZH", "JP"]
    )

    args = parser.parse_args()
    if args.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(args.config)

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else (
            "mps"
            if sys.platform == "darwin" and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model, net_g, None, skip_optimizer=True)

    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    create_audio(args, hps)
    # generated_audio = tts_fn_create(args.text,args.speaker,args.sdp_ratio,args.noise_scale,args.noise_scale_w,args.length_scale,args.language)
	#
    # # 保存生成的音频为.wav文件
    # output_path = "./output_audio.wav"  # 修改保存路径
    # wavfile.write(output_path, hps.data.sampling_rate, generated_audio)
    #
    # print(f"音频已保存到{output_path}")
