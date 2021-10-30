import math
from mido import Message
import mido
import time
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import random
import sys
import os
import csv

#############
# Before Starting the test, determine how many XIANG levels i.e. 需要测几条线
# for each 线, we need to add a new csv document in this folder
# suggested naming: xiang_xx_tester_xxxxx_pse_data.csv

#### Data used for training the ML model
if not os.path.exists('all_pairs.csv'):
    all_pairs_f = open('all_pairs.csv', 'w')
    all_pairs_f.write(
        'tester,pitch_1,vel_1,pitch_2,vel_2,relationship,ref_idx\n')
    all_pairs_f.close()

## Change the name when openning a new sheet
f = open('xiang_60_alan.csv', 'w')
f.write('pitch,velocity\n')
f.close()

# Change it for every tester
# Not important but might worth further comparison
tester_name = 'alan'

#Global Xiang Parameter i.e. the velocity when reference pitch is 69
XIANG = 60




## Tone Class defined by Qu Yang
class Tone:

    def __init__(self, pitch, vel):
        self.pitch = pitch
        self.vel = vel

    def change_velocity(self, new_vel):
        self.vel = new_vel

## SoundPlayer Class defined by Qu Yang
class SoundPlayer:

    def __init__(self, duration=1.2, pause=0.7):
        self.duration = duration
        self.pause = pause
        self.outport = mido.open_output()

    def play_sound(self, tone: Tone, dur=1.2):
        self.outport.send(Message('note_on', note=tone.pitch, velocity=tone.vel))
        time.sleep(dur)
        self.outport.send(Message('note_off', note=tone.pitch))

    def play_pair(self, ref_tone: Tone, var_tone: Tone, pause=0.7):

        time.sleep(1)
        # mode 1: play in random order
        idx = random.randint(0, 1)
        if idx:
            self.play_sound(var_tone)
            time.sleep(pause)

            self.play_sound(ref_tone)
            time.sleep(pause)
        else:
            self.play_sound(ref_tone)
            time.sleep(pause)

            self.play_sound(var_tone)
            time.sleep(pause)

        return idx

## TestManager Class defined by Qu Yang
class TestManager:

    def __init__(self, xiang_pitch, xiang, var_pitch, vel_min, vel_max):
        self.xiang = xiang
        self.var_pitch = var_pitch
        self.vel_list = np.linspace(vel_min, vel_max, vel_max - vel_min + 1)

        self.test_vel = []
        self.test_fb = []

        self.ref_stimuli = Tone(pitch=xiang_pitch, vel=xiang)  # based on the definition of xiang
        self.sound_player = SoundPlayer()

    def play_round(self, test_round):
        if test_round == 1:
            vel_max = self.xiang + 3 * 8
            is_var_louder = self.run_test(var_vel=vel_max)

            while not is_var_louder and vel_max <= 90:
                vel_max += 5
                is_var_louder = self.run_test(var_vel=vel_max)
            if not is_var_louder:
                print(f"No point of subject equality can be found for pitch {self.var_pitch} of {self.xiang} xiang")
                sys.exit(0)
        elif test_round == 2:
            vel_min = self.xiang - 3 * 8
            is_var_louder = self.run_test(var_vel=vel_min)

            while is_var_louder and vel_min >= 20:
                vel_min -= 5
                is_var_louder = self.run_test(var_vel=vel_min)
            if is_var_louder:
                print(f"No point of subject equality can be found for pitch {self.var_pitch} of {self.xiang} xiang")
                sys.exit(0)
        else:
            vel = self.get_var_velocity()[0]

            rand = random.randrange(-16, 16, 1)
            vel += rand

            # print(vel, rand)

            is_var_louder = self.run_test(var_vel=int(vel))
        return is_var_louder

    def run_test(self, var_vel):

        # handle corner case
        if var_vel < 30:
            var_vel = 30
        if var_vel > 90:
            var_vel = 90

        var_stimuli = Tone(pitch=self.var_pitch, vel=var_vel)
        ref_idx = self.sound_player.play_pair(self.ref_stimuli, var_stimuli)

        print("Please enter O if the former is louder, enter P if the latter is louder:")
        feedback = input()

        while feedback == '2':
            print("Replay")
            ref_idx = self.sound_player.play_pair(self.ref_stimuli, var_stimuli)
            print("Please enter O if the former is louder, enter P if the latter is louder:")
            feedback = input()


        feedback = feedback.strip()
        # temp transposing key inputs
        feedback = key_map(feedback)

        # classification, 0 for ref is louder, 1 for var is louder
        if int(feedback) == ref_idx:
            is_var_louder = 0
        else:
            is_var_louder = 1

        self.test_vel.append([var_vel])
        self.test_fb.append(is_var_louder)


        # Write a pair of > or <
        with open(r'all_pairs.csv', 'a') as all_pairs:
            writer = csv.writer(all_pairs)
            if not ref_idx:
                writer.writerow([tester_name, self.ref_stimuli.pitch, self.ref_stimuli.vel, self.var_pitch, var_vel, is_var_louder ^ ref_idx, ref_idx])
            else:
                writer.writerow([tester_name, self.var_pitch, var_vel, self.ref_stimuli.pitch, self.ref_stimuli.vel,
                                  is_var_louder ^ ref_idx, ref_idx])
        return is_var_louder

    def get_var_velocity(self):
        # fit curve
        lgc = LogisticRegression(random_state=0).fit(self.test_vel, self.test_fb)

        prob_all = lgc.predict_proba(self.vel_list.reshape(len(self.vel_list), 1))

        # probability of var is louder than ref
        prob_var = np.array(list(map(lambda p: p[1], prob_all)))

        relative_prob_var = abs(prob_var - 0.5)
        pse_idx = np.where(relative_prob_var == min(relative_prob_var))[0][0]

        pse = self.vel_list[pse_idx]

        s_ave = 0
        for v in self.test_vel:
            p = prob_var[v[0] - 30] # p need to p > 1
            try: 
                s = (pse - v) / math.log(1/p - 1, math.e)
            except: 
                pass
            else:
                s_ave += s

        s_ave = s_ave / len(self.test_vel)
        sd = (s_ave * math.pi) / math.sqrt(3)  # standard deviation

        # self.draw_plot(prob_var, pse, pse_idx)

        return pse, s, sd

    def draw_plot(self, prob, pse, pse_idx):
        plt.title(f'Fitted Logistic function for round {len(self.test_vel) - 2}')
        plt.xlabel("midi velocity")
        plt.ylabel("Prob that var is louder than ref")

        x = self.vel_list
        y = prob

        x_var = []
        y_var = []
        x_ref = []
        y_ref = []
        for idx, v in enumerate(self.test_vel):
            if self.test_fb[idx]:
                x_var.append(v)
                y_var.append(y[int(v - min(x))])
            else:
                x_ref.append(v)
                y_ref.append(y[int(v - min(x))])

        plt.scatter(x_var, y_var, marker='o', label="var is louder")
        plt.scatter(x_ref, y_ref, marker='^', label="ref is louder")
        plt.scatter(pse, y[pse_idx], marker='*', c='r', label="estimated PSE")

        plt.plot(x, y)

        plt.legend(loc='upper left')
        plt.show()

# Key mapping: currently with O for smaller, P for larger (looking lowercase in terminal)
def key_map(key):
    M = {'o': '0', 'p': '1'}
    return M.get(key)

# Main program, does one note measurement
def main(var_pitch, ref_vel, ref_pitch=69, total_test_round=10):
    # set xiang
    xiang = ref_vel

    # set variable pitch
    # var_pitch = 21


    test_manager = TestManager(
        xiang_pitch = ref_pitch,
        xiang=xiang,
        var_pitch=var_pitch,
        vel_min=30,
        vel_max=90,
    )

    test_round = 0
    while test_round < total_test_round + 2:
        test_round += 1
        test_manager.play_round(test_round)

    pse, s, sd = test_manager.get_var_velocity()

    print(f"The point of subjective equality for pitch {var_pitch} of {xiang} xiang is: {pse}, sd is: {s}")
    return pse

def recursiveRefLink(pitch, direction):
    '''
    recursively compute all the reference pitches PSE velocity and write them into the dataframe, 
    then start the normal test.
    pitch: the changing reference pitch
    direction: the pitch is either going up or going down, i.e. 1 or -1
    '''
    if pitch < 21 or pitch > 109:
        return
    else:
        one_octive_compare = main(
            pitch, XIANG, ref_pitch=pitch-12*direction, total_test_round=10)
        two_octive_compare = one_octive_compare
        if direction * (pitch - 69) >= 24:
            two_octive_compare = main(
                pitch, XIANG, ref_pitch=pitch-24*direction, total_test_round=10)
        compare_average = (one_octive_compare + two_octive_compare) / 2
        all_ref_pitches.at[(pitch - 21) // 12, 'pse_velocity'] = compare_average
        recursiveRefLink(pitch+direction*12, direction)

# Running the changing reference measurement
if not os.path.exists('all_ref_pitches.csv'):
    # A pandas dataframe to store (pitch, PSE velocity) values for changed ref pitches
    # Pitches are All MIDI 'A's i.e. [21,33,45,57,69,81,93,105]
    all_ref_pitches = pd.DataFrame([[p, 0] for p in range(
        21, 109, 12)], columns=['pitch', 'pse_velocity'])
    all_ref_pitches.at[4, 'pse_velocity'] = XIANG
    recursiveRefLink(57, -1)
    recursiveRefLink(81, 1)
    print(all_ref_pitches)
    all_ref_pitches.to_csv('all_ref_pitches.csv', index=False)


# Doing with ranges for changed reference pitch
if not os.path.exists('progress.npy'):
    all_ref_measured = []
    progress_array = []
    with open('all_ref_pitches.csv') as csvfile:
        Dreader = csv.DictReader(csvfile)
        for row in Dreader:
            all_ref_measured.append(row)
    for ref in all_ref_measured:
        ref_p = int(ref['pitch'])
        range_low = ref_p - 7
        range_high = ref_p + 7
        if ref_p == 21:
            range_low = 21
        elif ref_p == 105:
            range_high = 109
        ref['range'] = range(range_low, range_high+1)
        progress_array += [[ref_p, int(ref['pse_velocity']), var, 0] for var in range(range_low, range_high+1)]
    np.save('progress.npy', progress_array)
else:
    progress_array = np.load('progress.npy')


#################################
### actual program            ###
#################################
#################################

for unit in progress_array:
    '''
    unit: array[4] , [0]: ref pitch, [1]: ref velocity, [2]: variable pitch, [3]: status bit
    '''
    print(unit)
    ref_p = unit[0]
    ref_v = unit[1]
    var_p = unit[2]
    # To Debug: comment out the line below
    #####################################
    ############ main line ##############
    pse_var_p = main(var_p, ref_v, ref_p)
    #####################################
    # for every pair done, write the result to the csv file
    # when finished, this csv file contains 一根线
    # CHANGE THE FILE NAME when 测另一根线
    with open(r'xiang_60_alan.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([var_p, pse_var_p])






#### This part does not matter

# x = list(range(30, 90))
# y = list(map(lambda p: 1 / (1 + pow(math.e, - (p - 53) / 1.58)), x))
# plt.plot(x, y)
# plt.show()
