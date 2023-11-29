import os
model_name = 'gxd'
out_file = f"filelists/{model_name}.list" #此处更改人名（英文简称）

def process():
    with open(out_file,'w' , encoding="Utf-8") as wf:
        ch_name = '{model_name}' #此处更改人名（英文简称）
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
