import numpy as np
import librosa
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DS_FACTOR = 8
EPS = 0.4
MIN_E = 0.05
MIN_LAG = 2
MAX_LAG = 110
N = 8
LMIN_OFFSET = 0
e_cache = np.zeros(N)
h_cache = np.zeros(N)


def main():
    def ds_predict_pitch(idx):
        frame = np.zeros(128)

        lmin0 = 0
        lmin1 = 0
        lmin1_conf = 1000
        lmin2 = 0
        lmin2_conf = 1000
        lmin = 0

        troughs = []
        initial_peak = -1
        rising_edge = -1
        falling_edge = -1
        confidence_prev_prev = 0
        confidence_prev = 0
        confidence = (ds_e_cache[2] - 2 * ds_h_cache[2]) * ds_e_cache[2]
        confidence_next = (ds_e_cache[2 + 1] -
                           (2 * ds_h_cache[2 + 1])) * ds_e_cache[2 + 1]
        confidence_next_next = (ds_e_cache[2 + 2] -
                                (2 * ds_h_cache[2 + 2])) * ds_e_cache[2 + 2]
        for lag in range(MIN_LAG, MAX_LAG):
            if initial_peak < 0 and confidence > confidence_prev_prev and confidence >= confidence_prev and confidence >= confidence_next and confidence > confidence_next_next:
                initial_peak = lag

            if ds_e_cache[lag] > MIN_E:
                if initial_peak > 0:
                    if confidence < confidence_prev_prev and confidence <= confidence_prev and confidence <= confidence_next and confidence < confidence_next_next:
                        # we are in a local minima
                        if (ds_e_cache[lag] - (2 * ds_h_cache[lag])) < EPS:
                            # we are under the non-scaled threshold
                            if confidence < (ds_e_cache[lag] * EPS):
                                # we are under the scaled threshold
                                troughs.append(lag)
                                # if lag > MAX_LAG / 2:
                                #     break
                                # if len(troughs) == 2:
                                #     # early break if we have found 2 x lmin values
                                #     break

            frame[lag] = confidence
            confidence_prev_prev = confidence_prev
            confidence_prev = confidence
            confidence = confidence_next
            confidence_next = confidence_next_next
            confidence_next_next = (
                ds_e_cache[lag + 2] -
                (2 * ds_h_cache[lag + 2])) * ds_e_cache[lag + 2]

        # we can make this a lot more efficient by stopping early here
        # Also need to tidy up the several competing approaches
        if len(troughs) > 0:
            lmin1 = troughs[0]
            lmin1_conf = frame[lmin1]

            if len(troughs) > 1:
                lmin2 = troughs[1]
                lmin2_conf = frame[lmin2]

            # Here we check for a valid lmin2
            lmin2_is_valid = lmin2 > 0 and lmin2 < 110

            # and then compare the confidences (already scaled for the lag relative to E for a fair comparison)
            if lmin2_is_valid and (lmin2_conf * 3) < lmin1_conf:
                frame[lmin2] = lmin2_conf * 3
                lmin = lmin2
            else:
                frame[lmin2] = lmin2_conf * 3
                lmin = lmin1
        else:
            lmin1 = 0
            lmin1_conf = 0
            lmin2 = 0
            lmin2_conf = 0
            lmin = 0

        if ds_e_cache[lmin] < MIN_E:
            lmin = 0

        global LMIN_OFFSET
        LMIN_OFFSET = lmin * DS_FACTOR

        return frame, lmin1, lmin2, lmin, lmin1_conf, lmin2_conf, troughs

    def ds_update_e_and_h(idx,
                          min_lag=MIN_LAG,
                          max_lag=MAX_LAG,
                          ds_factor=DS_FACTOR):
        # wrap_ds = lambda x, idx: (idx - (x * DS_FACTOR)) % mask
        wrap_ds = lambda x, idx: (idx - (ds_factor * x)) % mask
        for lag in range(min_lag, max_lag):
            onelag_idx = wrap_ds(lag, idx)
            twolag_idx = wrap_ds(2 * lag, idx)

            oldest_e = np.square(ds_buf[twolag_idx])
            oldest_h = ds_buf[onelag_idx] * ds_buf[twolag_idx]

            e = np.square(ds_buf[idx])
            h = ds_buf[idx] * ds_buf[onelag_idx]

            ds_e_cache[lag] = ds_e_cache[lag] - oldest_e + e
            ds_h_cache[lag] = ds_h_cache[lag] - oldest_h + h

        return

    def init_e_and_h(idx):
        global e_cache
        global h_cache

        if LMIN_OFFSET < (MIN_LAG * DS_FACTOR):
            e_cache = np.zeros(N)
            h_cache = np.zeros(N)
            return

        wrap = lambda x, idx: int((idx - x) % mask)
        offset = lambda x: int((N / 2) + (x - LMIN_OFFSET) - 1)
        min_lag = int(LMIN_OFFSET - N / 2 + 1)
        max_lag = int(LMIN_OFFSET + N / 2)

        if idx > (2*max_lag) + 1:
            test_e = np.sum(np.square(buf[idx-max_lag-max_lag:idx]))
            test_h = np.sum(buf[idx-max_lag:idx] * buf[idx-max_lag-max_lag:idx - max_lag])

        # work out e and h for max_lag
        for x in range(0, max_lag + 1):
            e_cache[offset(max_lag)] += np.square(buf[wrap(
                x, idx)]) + np.square(buf[wrap(x + max_lag, idx)])
            h_cache[offset(max_lag)] += buf[wrap(x, idx)] * buf[wrap(x + max_lag,  idx)]

        # take oldest_e and oldest_h away to get the other lags
        for lag in range(max_lag - 1, min_lag - 1, -1):
            e_cache[offset(lag)] = e_cache[offset(lag + 1)] - np.square(buf[wrap(2 * (lag + 1), idx)])
            h_cache[offset(lag)] = h_cache[offset(lag + 1)] - (buf[wrap((lag + 1), idx)] * buf[wrap(2 * (lag + 1), idx)])
        return

    def update_e_and_h(idx):
        global e_cache
        global h_cache

        if LMIN_OFFSET < MIN_LAG:
            e_cache = np.zeros(N)
            h_cache = np.zeros(N)
            return

        wrap = lambda x, idx: int((idx - x) % mask)
        offset = lambda x: int((N / 2) + (x - LMIN_OFFSET))
        min_lag = int(LMIN_OFFSET - N / 2)
        max_lag = int(LMIN_OFFSET + N / 2) - 1

        for lag in range(min_lag, max_lag + 1):
            nolag_idx = wrap(0, idx)
            onelag_idx = wrap(lag, idx)
            twolag_idx = wrap(2 * lag, idx)

            oldest_e = np.square(buf[twolag_idx])
            oldest_h = buf[onelag_idx] * buf[twolag_idx]

            e = np.square(buf[nolag_idx])
            h = buf[nolag_idx] * buf[onelag_idx]

            e_cache[offset(lag)] = e_cache[offset(lag)] - oldest_e + e
            h_cache[offset(lag)] = h_cache[offset(lag)] - oldest_h + h
        return

    def shift_e_and_h_down(idx):
        global e_cache
        global h_cache
        global LMIN_OFFSET

        wrap = lambda x, idx: int((idx - x) % mask)

        for x in range(N-1, 0, -1):
            e_cache[x] = e_cache[x - 1]
            h_cache[x] = h_cache[x - 1]

        # TODO: Check that the logic around moving e and h is correct
        e_cache[0] = e_cache[1] - np.square(buf[wrap((LMIN_OFFSET - (N / 2) - 1) * 2, idx)])
        h_cache[0] = h_cache[1] - buf[wrap((LMIN_OFFSET - (N / 2) - 1) * 2, idx)] * buf[wrap((LMIN_OFFSET - (N / 2) - 1), idx)]

        LMIN_OFFSET += 1
        return

    def shift_e_and_h_up(idx):
        global e_cache
        global h_cache
        global LMIN_OFFSET

        wrap = lambda x, idx: int((idx - x) % mask)

        for x in range(0, N - 1):
            e_cache[x] = e_cache[x + 1]
            h_cache[x] = h_cache[x + 1]

        # TODO: this seems right but the values for e_cache don't match the value for E when calculated with
        # np.sum(np.square(buf[idx - int(2 * (LMIN_OFFSET + (N/2))):idx]))
        e_cache[N - 1] = e_cache[N - 2] + np.square(buf[wrap((LMIN_OFFSET + (N / 2)) * 2, idx)])
        h_cache[N - 1] = h_cache[N - 2] - buf[wrap((LMIN_OFFSET + (N / 2)) * 2, idx)] * buf[wrap((LMIN_OFFSET + (N / 2)), idx)]

        LMIN_OFFSET -= 1
        return


    def predict_fine_pitch(idx):
        conf_cache = (e_cache - (2 * h_cache))
        lmin = np.argmin(conf_cache)

        if conf_cache[lmin] > (e_cache[lmin] * EPS):
            return 0
        elif lmin == 0 or lmin == N-1:
            return 0
        elif e_cache[lmin] < MIN_E:
            return 0

        # parabolic interpolation of lmin to find pmin
        s0 = conf_cache[lmin - 1]
        s1 = conf_cache[lmin]
        s2 = conf_cache[lmin + 1]

        adjustment = (s2 - s0) / (2 * (2 * s1 - s2 - s0))

        if abs(adjustment) > 1:
            adjustment = 0

        pmin = lmin + adjustment + LMIN_OFFSET

        # shift E and H higher or lower
        if lmin < N / 2: # less than 4
            shift_e_and_h_down(idx)
        elif lmin >= (N / 2) + 1: # greater than 5
            shift_e_and_h_up(idx)

        return pmin

    # audio_path = "/Users/xavriley/Projects/q/test/audio_files/sin_440.wav"
    audio_path = "/Users/xavriley/Projects/AutotuneInPython/bobby.wav"
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    buf = np.zeros(2048)
    ds_buf = np.zeros(2048)
    writephase = 1024
    mask = 2048
    s2 = 0
    s4 = 0
    s8 = 0
    ds_e_cache = np.zeros(128)
    ds_h_cache = np.zeros(128)
    frame_output = []
    troughs_output = []
    lmin_output = [[], [], []]
    conf_output = [[], []]
    ds_wave_output = []
    fine_pitch_output = [[], []]
    frame = np.zeros(128)
    trough = np.array([])

    wrap = lambda x: x % mask

    for s_idx, s in enumerate(y):
        buf[writephase] = s

        if writephase % DS_FACTOR == 0:
            # downsampling step
            n = wrap(writephase - DS_FACTOR)
            ds2_1 = s2 + buf[n] / 2 + buf[wrap(n + 1)] / 4
            ds2_2 = buf[wrap(n + 1)] / 4 + buf[wrap(n + 2)] / 2 + buf[wrap(
                n + 3)] / 4
            ds2_3 = buf[wrap(n + 3)] / 4 + buf[wrap(n + 4)] / 2 + buf[wrap(
                n + 5)] / 4
            ds2_4 = buf[wrap(n + 5)] / 4 + buf[wrap(n + 6)] / 2 + buf[wrap(
                n + 7)] / 4
            s2 = buf[wrap(n + 7)] / 4

            ds4_1 = s4 + ds2_1 / 2 + ds2_2 / 4
            ds4_2 = ds2_2 / 4 + ds2_3 / 2 + ds2_4 / 4
            s4 = ds2_4 / 4

            ds8_1 = s8 + ds4_1 / 2 + ds4_2 / 4
            s8 = ds4_2 / 4

            # NB: we leave gaps in the ds_buf to allow for alignment with the full resolution
            # signal later on
            ds_buf[writephase] = ds8_1
            ds_wave_output.append(ds8_1)

        if writephase % 40 == 0:
            ds_update_e_and_h(writephase)


            frame, lmin1, lmin2, lmin, lmin1_conf, lmin2_conf, trough = ds_predict_pitch(
                writephase)

            init_e_and_h(writephase)
            fine_pitch_output[1].append((lmin * DS_FACTOR) + 20)
            lmin_output[0].append(lmin1)
            lmin_output[1].append(lmin2)
            lmin_output[2].append(lmin)
            conf_output[0].append(lmin1_conf)
            conf_output[1].append(lmin2_conf)
            frame_output.append(frame)
            troughs_output.append(trough)
        elif writephase % 8 == 0:
            lmin_output[0].append(0)
            lmin_output[1].append(0)
            lmin_output[2].append(0)
            conf_output[0].append(0)
            conf_output[1].append(0)
            frame_output.append(frame)
            troughs_output.append(trough)
            ds_update_e_and_h(writephase)
            update_e_and_h(writephase)
        elif writephase % 5 == 0:
            fine_pitch_output[1].append(None)
        else:
            update_e_and_h(writephase)

        if writephase % 5 == 0:
            fine_pitch = predict_fine_pitch(writephase)
            if fine_pitch > 0:
                fine_pitch_output[0].append(fine_pitch)
            else:
                fine_pitch_output[0].append(None)

        writephase = (writephase + 1) % mask

    plt.plot(fine_pitch_output[0], color='b')
    plt.plot(fine_pitch_output[1], color='r')

    plt.show()

    # fig, ax = plt.subplots()
    # def animate(i):
    #     ax.clear()
    #     ax.plot(frame_output[5000 + i])
    #     ax.set_xlim(0, 60)
    #     ax.set_ylim(0, EPS)
    #
    # # run the animation
    # ani = FuncAnimation(fig, animate, frames=2000, interval=50, repeat=False)
    #
    # plt.show()

    fig, axs = plt.subplots(2, sharex=False, sharey=False)
    plt.tight_layout()
    fr = 5207



    # axs[0].plot(frame_output[fr])

    # tmp_troughs = np.zeros(128)
    # if (len(troughs_output[fr]) > 0):
    #     tmp_troughs[troughs_output[fr]] = EPS
    # axs[0].plot(tmp_troughs)
    # axs[0].plot(lmin_output[0][fr], conf_output[0][fr])
    # axs[0].plot(lmin_output[1][fr], conf_output[1][fr])
    # print(f"chosen lmin: {lmin_output[2][fr]}")
    # # axs[0].set_xlim(0, 60)
    # axs[0].set_ylim(0, EPS)

    range_start = 0
    range_end = len(lmin_output[2])
    axs[1].pcolormesh(np.array(frame_output).T[:, range_start:range_end],
                      norm=colors.PowerNorm(gamma=0.5),
                      cmap='PuBu_r',
                      shading='auto')
    axs[1].plot(lmin_output[1][range_start:range_end], ',', color=colors.to_rgba('y', 0.5))
    axs[1].plot(lmin_output[0][range_start:range_end], ',', color=colors.to_rgba('g', 0.5))
    axs[1].plot(lmin_output[2][range_start:range_end], '.', color=colors.to_rgba('r', 0.75))
    # axs[1].plot(ds_wave_output)
    plt.savefig('bobby.png', dpi=300)
    import subprocess
    subprocess.run(["open", "-g", "bobby.png"])


if __name__ == "__main__":
    main()
