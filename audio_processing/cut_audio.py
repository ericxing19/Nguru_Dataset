from pydub import AudioSegment
import numpy as np
file_path = r"C:\Users\邢\Desktop\new_violin_audio\b2"
audio = AudioSegment.from_file(r"C:\Users\邢\Desktop\new_violin_audio\violin-b-2.wav", "wav")
audio_time = len(audio)  #获取待切割音频的时长，单位是毫秒
cut_parameters = np.arange(3,audio_time/1000,5)  #np.arange()函数第一个参数为起点，第二个参数为终点，第三个参数为步长（10秒）
start_time = int(0)  #开始时间设为0
########################根据数组切割音频####################
for t in cut_parameters:
    stop_time = int(t * 1000)
    # pydub以毫秒为单位工作
    audio_chunk = audio[start_time:stop_time] #音频切割按开始时间到结束时间切割
    audio_chunk.export(file_path+ "\\" + "b2-{}.wav".format(int(t/5)), format="wav")
    # 保存音频文件，t/10只是为了计数，根据步长改变。步长为5就写t/5
    start_time = stop_time - 1
    #开始时间变为结束时间前4s---------也就是叠加上一段音频末尾的4s
    print('finish')