import librosa
import librosa.display
import numpy as np
import librosa.display     # librosa's display module (for plotting features)
import matplotlib.pyplot as plt # matplotlib plotting functions
import matplotlib.style as ms
from numpy import diff
from scipy import interpolate
import math
import pandas as pd
from scipy.io.wavfile import read


targ_intv = [[0.1, 0.7], [0.3, 1.0], [0.7, 1.2], [1, 1.5]]

path = '/Users/yutianqin/Downloads/SchoolWork/Piano/recordings/'
dur = [0.1, 0.3, 0.7, 1]
vel = list(range(1,127,8))+[127]
sr = 48000


def time_vs_eng(wave, duration, p, d, v, e):
    human_ear = {0:0.512143375, 1: 0.6662215, 2:0.9921095, 3:1.311338875}
    name = e + '-pitch-'+str(p)+'vel-'+str(v)+'dur-'+str(d)

    C = np.abs(librosa.stft(wave, n_fft=16384, hop_length=256))
    freqs = librosa.fft_frequencies(48000, 16384)
    D_weighted = librosa.perceptual_weighting(C ** 2, freqs, ref=0.00002)
    D_weighted[D_weighted < 25] = 0

    librosa.display.specshow(D_weighted, y_axis='log', x_axis='time', sr=48000,
                             hop_length=256)
    plt.axvline(x=duration, color='g')
    # plt.axvline(x=human_ear[d], color='b')
    plt.title('spectrogram--' + name)
    plt.colorbar(format='%+2.0f dB')
    # plt.savefig("./" +'Weighted_spec-' + name + ".png")
    plt.show()

    ENG = []
    for i in range(len(D_weighted[0])):
        E = D_weighted[:, i:(i + 1)]
        erg = np.sum(E)
        ENG.append(erg)

    print('ENG here: ', ENG[int(duration/1.5*282)])
    print('MAX: ', max(ENG))


    plt.plot(ENG)
    plt.axvline(x=duration/1.5*282 , color='r', alpha=0.3)
    plt.title('Weighted-amp-' + name)
    # plt.savefig("./" + 'MWeighted-amp-' + name + ".png")
    plt.show()


def plotting(wave, duration, p, d, v, e):
    D = librosa.stft(wave, n_fft=16384, hop_length=256)

    name = e + '-pitch-'+str(p)+'vel-'+str(v)+'dur-'+str(d)

    # librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max), y_axis='log', x_axis='time', sr=48000, hop_length=256)
    # plt.axvline(x=duration , color='g')
    # plt.title('spectrogram--' + name)
    # plt.colorbar(format='%+2.0f dB')
    # # plt.savefig("./" +'MAX_spec-' + name + ".png")
    # plt.show()

    print('##### duration = ', duration, '#####')
    #     time = np.linspace(0, 1.5, len(wave))
    plt.plot(wave)
    plt.axvline(x=duration * 48000, color='r', alpha=0.3)
    plt.title('wave--' + name)
    # plt.savefig("./" + 'Wave-' + name + ".png")
    plt.show()



def loading(pitch, vel_level, dur_level, notetype):

    name = 'dictwave_' + notetype + '_pitch-' + str(pitch) + '_vel-' + str(vel[vel_level]) + \
           '_dur-' + str(dur[dur_level]) + '_idx-' + str(dur_level + 4 * vel_level + 4 * len(vel) * (pitch - 21))

    wave, sr = librosa.load(path + name + '.wav', sr=48000)

    return wave


##Onset
def findT0(wave, hop_len=256):
    # def findT0(wave, hop_len=512):
    # # maoran's method
    # energy = np.array(wave ** 2)
    # onset = np.where(energy == max(energy[:5000]))[0][0]  # maximun energy Paseval's Theorem

    # # roger's method
    W = {}

    for j in range(0, 42):
        W[j] = []
        for i in range(0, 1050):
            W[j].append(wave[i + 1050 * j])

    rms_all = []

    for j in range(0, 42):
        w_power = np.array(np.array(W[j]) ** 2)
        rms = np.sqrt(np.sum(w_power) / 1050)
        rms_all.append(rms)

    rms_all = np.array(rms_all)
    onset = np.where(rms_all == np.max(rms_all))[0][0] * 1050

    # plt.plot(wave)
    # plt.axvline(x=onset, c='r', alpha=0.5)
    # plt.show()

    t0 = librosa.samples_to_frames(onset, hop_length=hop_len)
    # t0 = 5

    t0 = t0 / (sr / hop_len)

    return t0


def get_D_matrices(wave, ref_value=1e-3):
    D = librosa.stft(wave, n_fft=16384, hop_length=256)  # set n_fft = 16384,
    Ddb = librosa.amplitude_to_db(np.abs(D), ref=ref_value)

    return D, Ddb


# midi pitch
def calc_overtone_index(p, bin_width):
    pitch = p - 20
    fund_freq = (2 ** ((pitch - 49) / (12))) * 440
    max_freq = (2 ** ((88 - 49) / (12))) * 440
    overtone = [fund_freq]
    i = 2
    while (fund_freq * i < max_freq):
        overtone.append(fund_freq * i)
        i += 1

    indices = []
    for f in overtone:
        if round(f / bin_width) < 8193:  # len(D) = 8193
            indices.append(round(f / bin_width))

    print('calc_overtone_index', overtone, indices)

    return overtone, indices

def polt_Aweighting(wave, duration, p, d, v, e):
    # D_origin = get_D_matrices(wave)[1]  # dB
    #
    # D_weighted = librosa.A_weighting(D_origin, -75.75632)

    C = np.abs(librosa.stft(wave, n_fft=16384, hop_length=256))
    freqs = librosa.fft_frequencies(48000, 16384)
    D_weighted = librosa.perceptual_weighting(C ** 2, freqs, ref = 0.00002)
    D_weighted[D_weighted < 25] = 0
    name = e + '-pitch-' + str(p) + 'vel-' + str(v) + 'dur-' + str(d)

    librosa.display.specshow(D_weighted, y_axis='log', x_axis='time', sr=48000, hop_length=256)
    plt.axvline(x=duration, color='g')
    plt.title('A_spectrogram--' + name)
    plt.colorbar(format='%+2.0f dB')
    # plt.savefig("./" +'MAX_spec-' + name + ".png")
    plt.show()

def plot_compare_curve(p, wave, no):
    D_origin = get_D_matrices(wave)[1]  # dB
    C = np.abs(librosa.stft(wave, n_fft=16384, hop_length=256))
    freqs = librosa.fft_frequencies(48000, 16384)
    D_weighted = librosa.perceptual_weighting(C ** 2, freqs, ref=0.00002)
    D_weighted[D_weighted < 25] = 0
    overtone, indices = calc_overtone_index(p, 48000 / 16384)

    #     print('plot_compare_curve', [len(D_origin[indices[i]]) for i in range(no)])

    return [D_weighted[indices[i]] for i in range(no)]



def find_d_and_weight(curve, low_b, up_b, onset=0.2, index = 2):
    dx = 1.5 / 282
    deriv = diff(curve) / dx
    deriv = deriv[int(low_b / dx): int(up_b / dx)]
    d = np.where(deriv == min(deriv))[0][0]
    D = low_b + dx * d
    D_index = int(low_b / dx) + d


    # deriv = curve[int(low_b / dx): int(up_b / dx)]
    # for i in range(int(low_b / dx), int(up_b / dx)):
    #     if curve[i] <= 0:
    #         D = low_b + dx * d
    #         D_index = int(low_b / dx) + d
    #         return max(low_b, D - onset), curve[D_index]

    #     plt.plot(curve)
    #     plt.axvline(x=D_index, color='r', alpha=0.3)
    #     plt.show()

    #     plt.plot(diff(curve) / dx, c = 'r', alpha = 0.7)
    #     plt.axvline(x=int(low_b / dx), color='g', alpha=0.3)
    #     plt.axvline(x=int(up_b / dx), color='g', alpha=0.3)
    #     plt.axvline(x=D_index, color='r', alpha=0.3)
    #     plt.show()

    #     print('D = ', D, 'D_index = ', D_index)
    #     print('max', max(low_b, D - onset), ' Cueve[D_index]', curve[D_index])

    return max(low_b, D - onset), curve[D_index]


    # for i in range(int(low_b /1.5 * 282), int(up_b /1.5 * 282)):
    #     if curve[index][i] < 0:
    #         D = i/282*1.5
    #         return max(low_b, D - onset)

# def thi_dis():
#     pass

def compute_D(p, v, d, envir):
    wave = loading(p, v, d, envir)

    low_b, up_b = targ_intv[d][0], targ_intv[d][1]


    curves = plot_compare_curve(p, wave, 3)


    Ds, ws = [], []

    onset = findT0(wave)
    #     print('Onset = ', onset)

    for c in curves:
        D, w = find_d_and_weight(c, low_b, up_b, onset)
        Ds.append(D)
        ws.append(w)

    Duration = np.average(Ds, weights=ws)

    # Duration = min(Ds)

    # Duration = find_d_and_weight(curves, low_b, up_b, onset)

    time_vs_eng(wave, Duration, p, d, v, envir)
    plotting(wave, Duration, p, d, v, envir)
    # polt_Aweighting(wave, Duration, p, d, v, envir)
    return Duration


def compute_delay(notetype):
    delay_list = []
    for p in range(21, 109):
        for v in range(17):
            for d in range(4):
                delay = compute_D(p, v, d, notetype)
                delay_list.append(delay)

    return delay_list

def main():

    # for i in range(0, 4):
    #     compute_D(69, 8, i, 'audi'))


    for j in range(0, 17):
        compute_D(69, j, 2, 'audi')

    # for k in range(61, 89):
    #     compute_D(k, 10, 2, 'audi')
    #

if __name__ == "__main__":
    main()