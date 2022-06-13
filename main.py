import numpy as np
import librosa
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

DS_FACTOR = 8
EPS = 0.04

def main():
    def ds_predict_pitch(idx):
        frame = np.zeros(128)

        region0_end = -1
        region1_start = -1
        region1_end = -1
        region2_start = -1
        region2_end = -1
        lmin0 = 0
        lmin1 = 0
        lmin1_conf = 1
        lmin2 = 0
        lmin2_conf = 1

        confidence = e_cache - (2 * h_cache)
        idxs, data = find_peaks(-confidence, height=-EPS)
        confs = data['peak_heights']

        # for lag in range(2, 110):
        #     confidence = e_cache[lag] - 2*h_cache[lag]
        #     if confidence < EPS:
        #         frame[lag] = confidence

        #         if region0_end > 0 and region1_end < 0:
        #             if region1_start < 0:
        #                 region1_start = lag
        #             if confidence < lmin1_conf:
        #                 lmin1_conf = confidence
        #                 lmin1 = lag

        #         if region0_end > 0 and region1_end > 0 and region2_end < 0:
        #             if region2_start < 0:
        #                 region2_start = lag
        #             if confidence < lmin2_conf:
        #                 lmin2_conf = confidence
        #                 lmin2 = lag
        #     else:
        #         frame[lag] = 1
        #         if region0_end < 0:
        #             region0_end = lag - 1 
        #         elif region1_start > 0 and region1_end < 0:
        #             region1_end = lag - 1 
        #         elif region1_end > 0 and region2_start > 0 and region2_end < 0:
        #             region2_end = lag - 1

        # frame[region0_end] = 2
        # frame[region1_start] = 35
        # frame[region1_end] = 40
        # frame[region2_start] = 190
        # frame[region2_end] = 100

        frame[idxs[np.where(confs < EPS)]] = 20

        frame = confidence
        lmin1 = idxs[0] if len(idxs) > 0 else 0
        lmin2 = idxs[1] if len(idxs) > 1 else 0
        lmin1_conf = confs[0] if len(confs) > 0 else 0
        lmin2_conf = confs[1] if len(confs) > 1 else 0

        if lmin1_conf < lmin2_conf:
            lmin = lmin1
        else:
            lmin = lmin2


        return frame, lmin1, lmin2, lmin, lmin1_conf, lmin2_conf

    def ds_update_e_and_h(idx):
        wrap_ds = lambda x, idx: (idx - (x * DS_FACTOR)) % mask
        for lag in range(2,110):
            onelag_idx = wrap_ds(lag, idx)
            twolag_idx = wrap_ds(2 * lag, idx)

            oldest_e = np.square(ds_buf[twolag_idx])
            oldest_h = ds_buf[onelag_idx] * ds_buf[twolag_idx]

            e = np.square(ds_buf[idx])
            h = ds_buf[idx] * ds_buf[onelag_idx]

            e_cache[lag] = e_cache[lag] - oldest_e + e
            h_cache[lag] = h_cache[lag] - oldest_h + h

            if lag == 12:
                slow_e = 0
                slow_h = 0
                print(twolag_idx)
                print(idx)
                for i in range(0, 2 * lag):
                    slow_e += np.square(ds_buf[wrap_ds(i, idx)])
                for i in range(0, lag):
                    slow_h += (ds_buf[wrap_ds(i, idx)] * ds_buf[wrap_ds(lag + i, idx)])

                if(e_cache[lag] != slow_e):
                    breakpoint()

                print(f"E {e_cache[lag] - slow_e}")
                print(f"H {h_cache[lag] - slow_h}")
        return

    audio_path = "/home/xavriley/Projects/q/test/audio_files/sin_440.wav" 
    # audio_path = "/home/xavriley/Projects/AutotuneInPython/bobby.wav" 
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    buf = np.zeros(2048)
    ds_buf = np.zeros(2048)
    writephase = 1024
    mask = 2048
    s2 = 0
    s4 = 0
    s8 = 0
    e_cache = np.zeros(128)
    h_cache = np.zeros(128)
    frame_output = []
    lmin_output = [[], [], []]
    conf_output = [[], []]
    ds_wave_output = []


    for s_idx, s in enumerate(y):
        buf[writephase] = s

        # downsampling step
        n = writephase - DS_FACTOR
        ds2_1 = s2            + buf[n]/2   + buf[n+1]/4
        ds2_2 = buf[n+1]/4 + buf[n+2]/2 + buf[n+3]/4
        ds2_3 = buf[n+3]/4 + buf[n+4]/2 + buf[n+5]/4
        ds2_4 = buf[n+5]/4 + buf[n+6]/2 + buf[n+7]/4
        s2 = buf[n+7]/4

        ds4_1 = s4 + ds2_1/2 + ds2_2/4
        ds4_2 = ds2_2/4 + ds2_3/2 + ds2_4/4
        s4 = ds2_4/4

        ds8_1 = s8 + ds4_1/2 + ds4_2/4
        s8 = ds4_2/4
        ds_buf[writephase] = ds8_1
        ds_wave_output.append(ds8_1)

        if writephase % 40 == 0:
            # TODO: compare the non-circular buffered output with the linear
            # downsampled output to see if the circular buffer is causing the funny dips
            if s_idx/8 > 24: 
                lag = 12
                slow_e = 0
                slow_h = 0
                for i in range(0, 2 * lag):
                    slow_e += np.square(ds_wave_output[-i - 1])
                for i in range(0, lag):
                    slow_h += (ds_wave_output[-i - 1] * ds_wave_output[-i - 1 - lag])

                if(e_cache[lag] != slow_e):
                    breakpoint()

                print(f"E {e_cache[lag] - slow_e}")
                print(f"H {h_cache[lag] - slow_h}")

            ds_update_e_and_h(writephase)
            frame, lmin1, lmin2, lmin, lmin1_conf, lmin2_conf = ds_predict_pitch(writephase)
            lmin_output[0].append(lmin1)
            lmin_output[1].append(lmin2)
            lmin_output[2].append(lmin)
            conf_output[0].append(lmin1_conf)
            conf_output[1].append(lmin2_conf)
            frame_output.append(frame)
        elif writephase % 8 == 0:
            ds_update_e_and_h(writephase)

        writephase = (writephase + 1) % mask 

    fig, axs = plt.subplots(3)
    plt.tight_layout()
    axs[0].plot(conf_output[0])
    axs[0].plot(conf_output[1])
    axs[1].pcolormesh(np.array(frame_output).T, norm=colors.PowerNorm(gamma=0.5),
                       cmap='PuBu_r', shading='auto')
    axs[2].plot(lmin_output[0])
    axs[2].plot(lmin_output[1])
    axs[2].plot(lmin_output[2], '.')
    plt.show()

if __name__=="__main__":
    main()
