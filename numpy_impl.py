import numpy as np
from librosa import load, resample
from numba import jit
from os.path import exists
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


@jit(nopython=True)
def find_conf(y, ds_factor=1):
    eps = 0.4

    max_lag = int(880/ds_factor)
    min_lag = int(16/ds_factor)

    e_cache = np.zeros((max_lag, int(y.shape[0] / 5)))
    h_cache = np.zeros_like(e_cache)

    for lag in range(min_lag, max_lag):
        e: float = 0
        h: float = 0
        oldest_e: float = 0
        oldest_h: float = 0
        for (index,), x in np.ndenumerate(y):
            if index + (2 * max_lag) < y.shape[0]:
                e = e - oldest_e + np.square(y[index + (2 * lag - 1)])
                h = h - oldest_h + (y[index+lag] * y[index+((2*lag)-1)])
                oldest_e = np.square(y[index])
                oldest_h = y[index] * y[index+lag-1]
                if index % 5 == 0 and (index/5 < e_cache.shape[1]):
                    # print(f"{index} {lag}")
                    e_cache[lag, int(index/5)] = e
                    h_cache[lag, int(index/5)] = h

    return e_cache, h_cache

# @jit(nopython=False)
def find_lmin(res, e_cache):
    lmins = []
    lmin_idxs = []
    min_e = 0.5
    eps = 0.4
    min_lag = 16

    for idx in range(0, res.shape[1]):
        lmin = 0
        lmin1 = 1000
        lmin2 = 1000
        lmin_idx = 0

        frame = res[:, idx]
        e_frame = e_cache[:, idx]

        region_mins = []
        last_conf = 0
        region_start = -1
        region_end = -1
        for (lag,), conf in np.ndenumerate(frame):
            if conf > e_frame[lag]:
                conf = 0

            if last_conf == 0 and conf > 0:
                region_start = lag

            if region_start > -1:
                if last_conf >= 0 and conf == 0:
                    region_end = lag

            last_conf = conf

            if region_start > 0 and region_end > 0:
                if region_start >= region_end:
                    # skip zero length regions or regions in progress
                    continue

                if region_start == min_lag:
                    # skip any spurious minima near zero
                    continue

                # find the minima for this region
                current_lmin_idx = region_start + np.argmin(frame[region_start:region_end])

                if current_lmin_idx > lmin_idx:
                    lmin_idx = current_lmin_idx
                else:
                    continue

                if frame[lmin_idx] > (e_frame[lmin_idx] * eps):
                    continue

                if e_frame[lmin_idx] < min_e:
                    # if we found a local minima that didn't meet the min energy,
                    # then we store a zero prediction to prevent the larger lags
                    # from winning, by virtue of their energy tailing off more slowly
                    region_mins.append([0, 0])
                    continue

                region_mins.append([lmin_idx, frame[int(lmin_idx)]])
                region_start = -1
                region_end = -1

        if len(region_mins) == 0:
            lmins.append(0)
            lmin_idxs.append(0)
            continue

        if len(region_mins) > 0:
            lmin1 = region_mins[0][1]

        if len(region_mins) > 1:
            lmin2 = region_mins[1][1]

        if lmin2 and lmin2 < lmin1:
            lmin = lmin2
            lmin_idx = region_mins[1][0]
        elif lmin1 < 1000:
            lmin = lmin1
            lmin_idx = region_mins[0][0]

        if e_frame[int(lmin_idx)] < min_e:
            lmins.append(0)
            lmin_idxs.append(0)
            continue

        s0 = frame[lmin_idx - 1]
        s1 = frame[lmin_idx]
        s2 = frame[lmin_idx + 1]

        adjustment = (s2 - s0) / (2 * (2 * s1 - s2 - s0))

        if abs(adjustment) > 1:
            adjustment = 0

        pmin = lmin_idx + adjustment

        lmins.append(lmin)
        lmin_idxs.append(pmin)

    return lmins, lmin_idxs

if __name__ == '__main__':
    if not exists('cache.npy'):
        y, sr = load('bobby.wav', sr=None)
        y = np.pad(y, ((2*880)+1,0))

        y_8ds = resample(y, orig_sr=sr, target_sr=sr/8)
        print(y.shape)
        print(y_8ds.shape)

        e_cache, h_cache = find_conf(y)
        # e_cache, h_cache = find_conf(y_8ds, ds_factor=8)
        res = e_cache - (2 * h_cache)
        np.save('e_cache.npy', e_cache)
        np.save('h_cache.npy', h_cache)
        np.save('cache.npy', res)
    else:
        e_cache = np.load('e_cache.npy')
        h_cache = np.load('h_cache.npy')
        res = np.load('cache.npy')

    section_start = 0
    section_end = res.shape[1]
    res_chunk = res[:, section_start:section_end]
    e_chunk = e_cache[:, section_start:section_end]
    lmins, lmin_idxs = find_lmin(res_chunk, e_chunk)

    eps = 0.4
    fr = int(34500/8)
    frame = res[:, fr]

    # plt.plot(frame)
    # # plt.ylim(0, 1)
    # # plt.xlim(0, 50)
    # plt.plot(e_cache[:, fr] * eps)
    # # plt.vlines(troughs, 0, 20)
    # plt.plot(lmin_idxs[fr], lmins[fr], '.')
    # plt.show()

    def view_frame(f):
        return res_chunk[:, int(f)]

    def view_e_cache(f):
        return e_chunk[:, int(f)] * eps

    def view_lmins(f):
        return lmin_idxs[int(f)], lmins[int(f)]

    def view_min_e(f):
        return lmin_idxs[int(f)], e_chunk[int(lmin_idxs[int(f)]), int(f)]


    fig, ax = plt.subplots()
    ax.pcolormesh(res_chunk)
    ax.plot(lmin_idxs, ',')
    plt.savefig('bobby.png', dpi=300)
    import subprocess
    subprocess.run(["open", "-g", "bobby.png"])
    plt.clf()

    fig, ax = plt.subplots()
    ax.set_ylim(0, 20)
    initial_frame = int((section_end-section_start)/2)
    line, = plt.plot(view_frame(initial_frame), lw=2)
    e_line, = plt.plot(view_e_cache(initial_frame), lw=2)
    lmin_points, = plt.plot(*view_lmins(initial_frame), '.')
    min_e_points, = plt.plot(*view_min_e(initial_frame), '.')
    plt.hlines(0.1, 0, 880)

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(bottom=0.25)

    # Make a horizontal slider to control the frequency.
    # axfreq = plt.axes([0, 0.1, 5000, 880])
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider(
        ax=axfreq,
        label='',
        valmin=section_start,
        valmax=section_end,
        valinit=initial_frame,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        slider_val = frame_slider.val - section_start
        line.set_ydata(view_frame(slider_val))
        e_line.set_ydata(view_e_cache(slider_val))
        lmin_points.set_data(view_lmins(slider_val))
        min_e_points.set_data(view_min_e(slider_val))
        fig.canvas.draw_idle()


    # register the update function with each slider
    frame_slider.on_changed(update)
    plt.show()
