import os

filter=[".m4a"] #设置过滤后的文件类型 当然可以设置多个类型

def all_path(dirname):

    result = []#所有的文件

    for filename in os.listdir(dirname):
        apath = os.path.join(dirname, filename)  # 合并成一个完整路径
        ext = os.path.splitext(apath)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容

        if ext in filter:
            result.append(apath)

    return result

filenames=all_path(r"C:\Users\邢\Desktop\new_violin_audio")

for filename in filenames:
    filename=str(filename)
    temp=filename.split('.')
    print("success")

    #将.m4a格式转为wav格式的命令
    cmd_command = "ffmpeg -i {0} -acodec pcm_s16le -ac 1 -ar 16000 -y {1}.wav".format(filename,temp[0])
    # 将.mp3格式转为wav格式的命令
    #cmd_command = "ffmpeg -loglevel quiet -y -i {0} -ar 16000 -ac 1 {1}.wav && del {0}".format(filename, temp[0])

    #print(cmd_command)
    os.system(cmd_command)
