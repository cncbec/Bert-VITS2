# flake8: noqa: E402

import os
import logging

import re_matching
from tools.sentence import split_by_language

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

import torch
import utils
from infer import infer, latest_version, get_net_g, infer_multilang
import gradio as gr
import webbrowser
import numpy as np
from config import config
from tools.translate import translate

import requests
from pydub import AudioSegment
import argparse
from scipy.io import wavfile
net_g = None

device = config.webui_config.device
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def generate_audio(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            audio = infer(
                piece,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            # audio_list.append(silence)  # 将静音添加到列表中
    return audio_list


def generate_audio_multilang(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            audio = infer_multilang(
                piece,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language[idx],
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            # audio_list.append(silence)  # 将静音添加到列表中
    return audio_list


def tts_split(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    cut_by_sent,
    interval_between_para,
    interval_between_sent,
):
    if language == "mix":
        return ("invalid", None)
    while text.find("\n\n") != -1:
        text = text.replace("\n\n", "\n")
    para_list = re_matching.cut_para(text)
    audio_list = []
    if not cut_by_sent:
        for idx, p in enumerate(para_list):
            skip_start = idx != 0
            skip_end = idx != len(para_list) - 1
            audio = infer(
                p,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            silence = np.zeros((int)(44100 * interval_between_para), dtype=np.int16)
            audio_list.append(silence)
    else:
        for idx, p in enumerate(para_list):
            skip_start = idx != 0
            skip_end = idx != len(para_list) - 1
            audio_list_sent = []
            sent_list = re_matching.cut_sent(p)
            for idx, s in enumerate(sent_list):
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(sent_list) - 1) and skip_end
                audio = infer(
                    s,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                    sid=speaker,
                    language=language,
                    hps=hps,
                    net_g=net_g,
                    device=device,
                    skip_start=skip_start,
                    skip_end=skip_end,
                )
                audio_list_sent.append(audio)
                silence = np.zeros((int)(44100 * interval_between_sent))
                audio_list_sent.append(silence)
            if (interval_between_para - interval_between_sent) > 0:
                silence = np.zeros(
                    (int)(44100 * (interval_between_para - interval_between_sent))
                )
                audio_list_sent.append(silence)
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(
                np.concatenate(audio_list_sent)
            )  # 对完整句子做音量归一
            audio_list.append(audio16bit)
    audio_concat = np.concatenate(audio_list)
    return ("Success", (44100, audio_concat))


def tts_fn(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
):
    audio_list = []
    if language == "mix":
        bool_valid, str_valid = re_matching.validate_text(text)
        if not bool_valid:
            return str_valid, (
                hps.data.sampling_rate,
                np.concatenate([np.zeros(hps.data.sampling_rate // 2)]),
            )
        result = []
        for slice in re_matching.text_matching(text):
            _speaker = slice.pop()
            temp_contant = []
            temp_lang = []
            for lang, content in slice:
                if "|" in content:
                    temp = []
                    temp_ = []
                    for i in content.split("|"):
                        if i != "":
                            temp.append([i])
                            temp_.append([lang])
                        else:
                            temp.append([])
                            temp_.append([])
                    temp_contant += temp
                    temp_lang += temp_
                else:
                    if len(temp_contant) == 0:
                        temp_contant.append([])
                        temp_lang.append([])
                    temp_contant[-1].append(content)
                    temp_lang[-1].append(lang)
            for i, j in zip(temp_lang, temp_contant):
                result.append([*zip(i, j), _speaker])
        for i, one in enumerate(result):
            skip_start = i != 0
            skip_end = i != len(result) - 1
            _speaker = one.pop()
            idx = 0
            while idx < len(one):
                text_to_generate = []
                lang_to_generate = []
                while True:
                    lang, content = one[idx]
                    temp_text = [content]
                    if len(text_to_generate) > 0:
                        text_to_generate[-1] += [temp_text.pop(0)]
                        lang_to_generate[-1] += [lang]
                    if len(temp_text) > 0:
                        text_to_generate += [[i] for i in temp_text]
                        lang_to_generate += [[lang]] * len(temp_text)
                    if idx + 1 < len(one):
                        idx += 1
                    else:
                        break
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(one) - 1) and skip_end
                print(text_to_generate, lang_to_generate)
                audio_list.extend(
                    generate_audio_multilang(
                        text_to_generate,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        _speaker,
                        lang_to_generate,
                        skip_start,
                        skip_end,
                    )
                )
                idx += 1
    elif language.lower() == "auto":
        for idx, slice in enumerate(text.split("|")):
            if slice == "":
                continue
            skip_start = idx != 0
            skip_end = idx != len(text.split("|")) - 1
            sentences_list = split_by_language(
                slice, target_languages=["zh", "ja", "en"]
            )
            idx = 0
            while idx < len(sentences_list):
                text_to_generate = []
                lang_to_generate = []
                while True:
                    content, lang = sentences_list[idx]
                    temp_text = [content]
                    lang = lang.upper()
                    if lang == "JA":
                        lang = "JP"
                    if len(text_to_generate) > 0:
                        text_to_generate[-1] += [temp_text.pop(0)]
                        lang_to_generate[-1] += [lang]
                    if len(temp_text) > 0:
                        text_to_generate += [[i] for i in temp_text]
                        lang_to_generate += [[lang]] * len(temp_text)
                    if idx + 1 < len(sentences_list):
                        idx += 1
                    else:
                        break
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(sentences_list) - 1) and skip_end
                print(text_to_generate, lang_to_generate)
                audio_list.extend(
                    generate_audio_multilang(
                        text_to_generate,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        speaker,
                        lang_to_generate,
                        skip_start,
                        skip_end,
                    )
                )
                idx += 1
    else:
        audio_list.extend(
            generate_audio(
                text.split("|"),
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                speaker,
                language,
            )
        )

    audio_concat = np.concatenate(audio_list)
    return "Success", (hps.data.sampling_rate, audio_concat)

def tts_fn_create(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
):
    audio_list = []
    if language == "mix":
        bool_valid, str_valid = re_matching.validate_text(text)
        if not bool_valid:
            return str_valid, (
                hps.data.sampling_rate,
                np.concatenate([np.zeros(hps.data.sampling_rate // 2)]),
            )
        result = []
        for slice in re_matching.text_matching(text):
            _speaker = slice.pop()
            temp_contant = []
            temp_lang = []
            for lang, content in slice:
                if "|" in content:
                    temp = []
                    temp_ = []
                    for i in content.split("|"):
                        if i != "":
                            temp.append([i])
                            temp_.append([lang])
                        else:
                            temp.append([])
                            temp_.append([])
                    temp_contant += temp
                    temp_lang += temp_
                else:
                    if len(temp_contant) == 0:
                        temp_contant.append([])
                        temp_lang.append([])
                    temp_contant[-1].append(content)
                    temp_lang[-1].append(lang)
            for i, j in zip(temp_lang, temp_contant):
                result.append([*zip(i, j), _speaker])
        for i, one in enumerate(result):
            skip_start = i != 0
            skip_end = i != len(result) - 1
            _speaker = one.pop()
            idx = 0
            while idx < len(one):
                text_to_generate = []
                lang_to_generate = []
                while True:
                    lang, content = one[idx]
                    temp_text = [content]
                    if len(text_to_generate) > 0:
                        text_to_generate[-1] += [temp_text.pop(0)]
                        lang_to_generate[-1] += [lang]
                    if len(temp_text) > 0:
                        text_to_generate += [[i] for i in temp_text]
                        lang_to_generate += [[lang]] * len(temp_text)
                    if idx + 1 < len(one):
                        idx += 1
                    else:
                        break
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(one) - 1) and skip_end
                print(text_to_generate, lang_to_generate)
                audio_list.extend(
                    generate_audio_multilang(
                        text_to_generate,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        _speaker,
                        lang_to_generate,
                        skip_start,
                        skip_end,
                    )
                )
                idx += 1
    elif language.lower() == "auto":
        for idx, slice in enumerate(text.split("|")):
            if slice == "":
                continue
            skip_start = idx != 0
            skip_end = idx != len(text.split("|")) - 1
            sentences_list = split_by_language(
                slice, target_languages=["zh", "ja", "en"]
            )
            idx = 0
            while idx < len(sentences_list):
                text_to_generate = []
                lang_to_generate = []
                while True:
                    content, lang = sentences_list[idx]
                    temp_text = [content]
                    lang = lang.upper()
                    if lang == "JA":
                        lang = "JP"
                    if len(text_to_generate) > 0:
                        text_to_generate[-1] += [temp_text.pop(0)]
                        lang_to_generate[-1] += [lang]
                    if len(temp_text) > 0:
                        text_to_generate += [[i] for i in temp_text]
                        lang_to_generate += [[lang]] * len(temp_text)
                    if idx + 1 < len(sentences_list):
                        idx += 1
                    else:
                        break
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(sentences_list) - 1) and skip_end
                print(text_to_generate, lang_to_generate)
                audio_list.extend(
                    generate_audio_multilang(
                        text_to_generate,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        speaker,
                        lang_to_generate,
                        skip_start,
                        skip_end,
                    )
                )
                idx += 1
    else:
        audio_list.extend(
            generate_audio(
                text.split("|"),
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                speaker,
                language,
            )
        )

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
        "-s", "--speaker", default="gaoxiangdong", type=str, help="speaker short name"
    )
    parser.add_argument(
        "-r", "--sdp_ratio", default=0.2, action="store_true", help="sdp ratio"
    )
    parser.add_argument(
        "--noise_scale", default=0.6, action="store_true", help="noise scale" #minimum=0.1, maximum=2
    )
    parser.add_argument(
        "--noise_scale_w", default=0.8, action="store_true", help="noise scale w" #minimum=0.1, maximum=2
    )
    parser.add_argument(
        "--length_scale", default=1.0, action="store_true", help="length scale" #minimum=0.1, maximum=2
    )
    parser.add_argument(
        "--language", default="auto", action="store_true", help="language" #["ZH", "JP", "EN", "mix", "auto"]
    )

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    # 若config.json中未指定版本则默认为最新版本
    version = hps.version if hasattr(hps, "version") else latest_version
    net_g = get_net_g(
        model_path=config.webui_config.model, version=version, device=device, hps=hps
    )
    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["ZH", "JP", "EN", "mix", "auto"]


    # create_audio(args, hps)
    generated_audio = tts_fn_create(args.text,args.speaker,args.sdp_ratio,args.noise_scale,args.noise_scale_w,args.length_scale,args.language)
    # 将音频数据转换为16位有符号整数格式
    generated_audio = (generated_audio * (2 ** 15 - 1)).astype(np.int16)
    # 保存生成的音频为.wav文件
    output_path = "./output_audio.wav"  # 修改保存路径
    print('采样率是{}'.format(hps.data.sampling_rate))
    wavfile.write(output_path, hps.data.sampling_rate, generated_audio)

    print(f"音频已保存到{output_path}")
