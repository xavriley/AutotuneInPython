import numpy as np
import librosa
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

DS_FACTOR = 8
EPS = 0.04
MIN_E = 0.05


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
        lmin1_conf = 1000
        lmin2 = 0
        lmin2_conf = 1000
        lmin = 0

        # confidence = e_cache - (2 * h_cache)
        # idxs, data = find_peaks(-confidence, height=-EPS)
        # confs = data['peak_heights']

        # lmin1 = idxs[0] if len(idxs) > 0 and e_cache[idxs[0]] > MIN_E else 0
        # lmin2 = idxs[1] if len(idxs) > 1 and e_cache[idxs[1]] > MIN_E else 0
        # lmin1_conf = confs[0] if len(confs) > 0 and e_cache[
        #     idxs[0]] > MIN_E else 0
        # lmin2_conf = confs[1] if len(confs) > 1 and e_cache[
        #     idxs[0]] > MIN_E else 0

        troughs = []
        confidence_prev = 0
        confidence = (e_cache[2] - 2 * h_cache[2]) * e_cache[2]
        confidence_next = (e_cache[2 + 1] -
                           (2 * h_cache[2 + 1])) * e_cache[2 + 1]
        for lag in range(2, 110):
            # We need to avoid the lag that occasionally appears at zero
            if confidence > (e_cache[lag] * EPS) and region0_end < 0:
                region0_end = lag - 1

            if e_cache[lag] < MIN_E:
                continue

            # TODO: deal with flat local minima a == b == c, without picking up noise in the initial ramp up
            if confidence < confidence_prev and confidence < confidence_next:
                # we are in a local minima
                if (e_cache[lag] - (2 * h_cache[lag])) < EPS:
                    # we are under the non-scaled threshold
                    if confidence < (e_cache[lag] * EPS):
                        # we are under the scaled threshold
                        troughs.append(lag)

            # if confidence <= (e_cache[lag] * EPS):
            #     if region0_end > 0 and region1_end < 0:
            #         if region1_start < 0:
            #             region1_start = lag
            #         if confidence < lmin1_conf:
            #             lmin1_conf = confidence
            #             lmin1 = lag

            #     if region0_end > 0 and region1_end > 0 and region2_end < 0:
            #         if region2_start < 0:
            #             region2_start = lag
            #         if confidence < lmin2_conf:
            #             lmin2_conf = confidence
            #             lmin2 = lag
            # else:
            #     if region1_start > 0 and region1_end < 0:
            #         region1_end = lag - 1
            #     elif region1_end > 0 and region2_start > 0 and region2_end < 0:
            #         region2_end = lag - 1

            frame[lag] = confidence
            confidence_prev = confidence
            confidence = confidence_next
            confidence_next = (e_cache[lag + 1] -
                               (2 * h_cache[lag + 1])) * e_cache[lag + 1]

        # frame = (e_cache - (2 * h_cache)) * e_cache
        # frame[frame < e_cache * EPS] = 10

        # WIP: TODO: troughs calculation appears to be correct
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

        if e_cache[lmin] < MIN_E:
            lmin = 0

        # frame[region0_end] = 20
        # frame[region1_start] = 20
        # frame[region1_end] = 40
        # frame[region2_start] = 190
        # frame[region2_end] = 100

        # Highlight all areas that are candidates for a minLag
        # confidence = e_cache - (2 * h_cache)
        # frame[np.where(confidence <= (e_cache * EPS))] = 20

        return frame, lmin1, lmin2, lmin, lmin1_conf, lmin2_conf, troughs

    def ds_update_e_and_h(idx):
        # wrap_ds = lambda x, idx: (idx - (x * DS_FACTOR)) % mask
        wrap_ds = lambda x, idx: (idx - (DS_FACTOR * x)) % mask
        for lag in range(2, 110):
            onelag_idx = wrap_ds(lag, idx)
            twolag_idx = wrap_ds(2 * lag, idx)

            oldest_e = np.square(ds_buf[twolag_idx])
            oldest_h = ds_buf[onelag_idx] * ds_buf[twolag_idx]

            e = np.square(ds_buf[idx])
            h = ds_buf[idx] * ds_buf[onelag_idx]

            e_cache[lag] = e_cache[lag] - oldest_e + e
            h_cache[lag] = h_cache[lag] - oldest_h + h

            # if lag == 12:
            #     # print(f"circbuf {ds_buf[idx]} {ds_buf[twolag_idx+1]} {idx}")
            #     # if (abs(idx - 1024) / 8) >= 24:
            #     #     breakpoint()
            #     slow_e = 0
            #     slow_h = 0
            #     for i in range(0, 2 * lag):
            #         slow_e += np.square(ds_buf[wrap_ds(i, idx)])
            #     for i in range(0, lag):
            #         slow_h += (ds_buf[wrap_ds(i, idx)] *
            #                    ds_buf[wrap_ds(lag + i, idx)])

            #     # if (e_cache[lag] != slow_e):
            #     #     breakpoint()
            #     # print(f"Ecb: {e_cache[lag]} {abs(idx - 1024)}")

            #     # print(f"E {e_cache[lag] - slow_e}")
            #     # print(f"H {h_cache[lag] - slow_h}")
        return

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
    e_cache = np.zeros(128)
    h_cache = np.zeros(128)
    frame_output = []
    troughs_output = []
    lmin_output = [[], [], []]
    conf_output = [[], []]
    ds_wave_output = []
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

            # TODO: compare the non-circular buffered output with the linear
            # downsampled output to see if the circular buffer is causing the funny dips
            # if s_idx / 8 >= 24:
            #     lag = 12
            #     slow_e = 0
            #     slow_h = 0
            #     for i in range(0, (2 * lag)):
            #         slow_e += np.square(ds_wave_output[(-i - 1):][0])

            #     for i in range(0, lag):
            #         slow_h += (ds_wave_output[-i - 1] *
            #                    ds_wave_output[-i - 1 - lag])

            #     if e_cache[lag] != slow_e:
            #         print(f"Ediff {e_cache[lag] - slow_e}")
            #         print(f"Hdiff {h_cache[lag] - slow_h}")

            frame, lmin1, lmin2, lmin, lmin1_conf, lmin2_conf, trough = ds_predict_pitch(
                writephase)
            # if lmin1 < 8 and lmin > 0:
            #     breakpoint()
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

        writephase = (writephase + 1) % mask

    fig, axs = plt.subplots(2, sharex=False, sharey=False)
    plt.tight_layout()
    # axs[0].plot(conf_output[0], ',')
    # axs[0].plot(conf_output[1], ',')
    fr = 13048
    axs[0].plot(frame_output[fr])

    tmp_troughs = np.zeros(128)

    if (len(troughs_output[fr]) > 0):
        tmp_troughs[troughs_output[fr]] = EPS
    axs[0].plot(tmp_troughs)
    axs[0].plot(lmin_output[0][fr], conf_output[0][fr])
    axs[0].plot(lmin_output[1][fr], conf_output[1][fr])
    print(f"chosen lmin: {lmin_output[2][fr]}")
    # axs[0].set_xlim(0, 60)
    axs[0].set_ylim(0, EPS)
    axs[1].pcolormesh(np.array(frame_output).T,
                      norm=colors.PowerNorm(gamma=0.5),
                      cmap='PuBu_r',
                      shading='auto')
    axs[1].plot(lmin_output[1], ',', color=colors.to_rgba('y', 0.5))
    axs[1].plot(lmin_output[0], ',', color=colors.to_rgba('g', 0.5))
    axs[1].plot(lmin_output[2], '.', color=colors.to_rgba('r', 0.75))
    # axs[1].plot(ds_wave_output)
    plt.savefig('bobby.png', dpi=300)
    import subprocess
    subprocess.run(["open", "-g", "bobby.png"])


if __name__ == "__main__":
    main()
