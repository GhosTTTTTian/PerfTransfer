import mido
from mido import Message
import time
import numpy as np
import pyaudio
import random
import struct


def WhiteNoise(ref, var):
    if np.abs(ref - var) % 7 == 0:

        #钢琴洗脑
        outport = mido.open_output()
        # outport.send(Message('note_on', note=36, velocity=75))
        # outport.send(Message('note_on', note=42, velocity=75))
        # outport.send(Message('note_on', note=47, velocity=75))
        # outport.send(Message('note_on', note=51, velocity=75))
        # outport.send(Message('note_on', note=57, velocity=75))
        # outport.send(Message('note_on', note=61, velocity=75))
        outport.send(Message('note_on', note=69, velocity=75))
        # outport.send(Message('note_on', note=73, velocity=75))
        # outport.send(Message('note_on', note=79, velocity=75))
        # outport.send(Message('note_on', note=83, velocity=75))
        # outport.send(Message('note_on', note=89, velocity=75))
        time.sleep(2)
        outport.send(Message('note_off', note=69))

        #合成 white noise
        sr = 44100
        duration = 0.5

        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paInt16,
            channels= 1,
            rate= sr,
            input=False,
            output=True)

        for i in range(0, int(sr*duration)):

            output_bytes = struct.pack('h', 10*int(np.random.normal(0,256)))

            stream.write(output_bytes)

        stream.stop_stream()
        stream.close()
        p.terminate()

WhiteNoise(60, 67)