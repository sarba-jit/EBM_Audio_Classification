#!/usr/bin/python
'''
Author: SARBAJIT MUKHERJEE
Email: sarbajit.mukherjee@aggiemail.usu.edu

This code Breaks an 30 sec WAV audio segment into chunks that are <chunk_length> milliseconds long.
'''

import os
from pydub import AudioSegment
import pydub
import math
pydub.AudioSegment.ffmpeg = '/usr/bin/ffmpeg'


audio_path = 'Test_Data/whole_sounds/'
audio_path_chunked = 'Test_Data/chunked_sounds/'

def make_chunks(audio_segment, chunk_length):
    """
    Breaks an AudioSegment into chunks that are <chunk_length> milliseconds
    long.
    if chunk_length is 50 then you'll get a list of 50 millisecond long audio
    segments back (except the last one, which can be shorter)
    """
    number_of_chunks = math.ceil(len(audio_segment) / (float(chunk_length)/2))
    return [audio_segment[i * (chunk_length/2):(i * (chunk_length/2))+chunk_length]
            for i in range(int(number_of_chunks)-1)]



def audio_input_for_chunking(item):
    if item.endswith('.wav'):
        x = os.path.join(audio_path, item)
        myaudio = AudioSegment.from_file(x)
        chunk_length_ms = 2000  # pydub calculates in millisec
        chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of 1.8 sec
        audio_formatted = str.split(x, '/')
        audio_name = str.split(audio_formatted[2],'.')

        for i, chunk in enumerate(chunks):
            chunk_name = str(audio_name[0])+'_{0}.wav'.format(i)
            if i<30:
                chunk.export(audio_path_chunked+chunk_name, format="wav")