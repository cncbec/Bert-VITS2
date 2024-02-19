#slice
import os
model_name = 'tongyong'
out_file = f"filelists/{model_name}.list" #此处总的list文件名

def process():
  with open(out_file,'w' , encoding="Utf-8") as wf:
    folder_path = './data'
    # 遍历文件夹，获取wav的文件名，进行识别
    for file_name in os.listdir(folder_path):
      # 拼接文件的完整路径
      file_path = os.path.join(folder_path, file_name)
      # 检查是否是.wav文件
      if os.path.isfile(file_path) and file_name.endswith('.wav'):
        # 获取不带后缀的文件名
        file_name_without_extension = os.path.splitext(file_name)[0]
        ch_name = file_name_without_extension #此处人名（英文简称）
        ch_language = 'ZH'
        path = f"./raw/{ch_name}"
        files = os.listdir(path)
        for f in files:
          if f.endswith(".lab"):
           with open(os.path.join(path,f),'r', encoding="utf-8") as perFile:
              line = perFile.readline() 
              result = f"./dataset/{ch_name}/{f.split('.')[0]}.wav|{ch_name}|{ch_language}|{line}"
              wf.write(f"{result}\n")

if __name__ == "__main__":
    process()
