'''
This demonstates that the technique also
performs well for NOISY datasets!
Feel free to play around with different
noice levels:
'''
noise_level = 0.1

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# Here the data is trimmed and nomalized to make the math simpler:
np.set_printoptions(suppress=True, precision=7)
tsA, omegasA = np.load('Data_material.npy')
mask_keep = (4 < tsA) & (tsA < 190)
tsB = tsA[mask_keep]
omegasB = omegasA[mask_keep]
omegasOG = omegasB


noise = np.random.normal(0, 1, omegasB.size) * noise_level
omegasB = omegasB + noise

min_t, max_t = np.min(tsB), np.max(tsB)
scale_factor = max_t - min_t
# print(scale_factor)
tsC = (tsB - min_t) / scale_factor


# here we specify the number of terms, that is, "how many sine waves do we want"
a = 321
ns = np.arange(a)


# Here I compute c_reals and c_comps which are the initial values for all the rotating vectors (https://youtu.be/r6sGWTCMz2k?si=Yq7j15p0pLTBmmdM)
angles_ns = -2 * np.pi * ns[:, np.newaxis] * tsC
cos_things_ns = np.cos(angles_ns)
sin_things_ns = np.sin(angles_ns)
res_ns = cos_things_ns * omegasB
ims_ns = sin_things_ns * omegasB
c_reals = simpson(res_ns, x=tsC, axis=1)
c_comps = simpson(ims_ns, x=tsC, axis=1)


# Here we specify which data points we want it to produce a smooth curve through
start_time = 0.29
end_time = 0.344
num_of_evals = 5678
times = np.linspace(start_time, end_time, num_of_evals)

# here the points inside the time interval is computed for later use:
mask_in_time = (tsC > start_time) & (tsC < end_time)

# Here the complex numbers are converted into polar ones.
# since we know that the datas lie in the real axis, we can combine the negative
# and positive terms. since 0 is neigher neg or pos, we divide it by 2 to prevent it from
# being counted twice:
radii = np.sqrt(c_reals ** 2 + c_comps ** 2)  # the lengths of the vectors are computed
radii[0] = radii[0] / 2 
init_angles = np.arctan2(c_comps, c_reals)

# Here we compute each sine wave for each time value:
waves = radii[:, np.newaxis] * np.cos(
    2 * np.pi * ns[:, np.newaxis] * times + init_angles[:, np.newaxis]
    )

fig, ax = plt.subplots()
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

for i in ns:
    ax.cla()
    total_us = 2 * np.sum(waves[:int(i)+1], axis=0)
    ax.plot(tsC[mask_in_time], omegasB[mask_in_time], 'o', color='black', label='Noisy data')
    ax.plot(tsC[mask_in_time], omegasOG[mask_in_time], color='grey', label='OG data')
    ax.plot(times, total_us, color='C0', label='Aproximation')
    plt.title(f"Terms: {int(i)}")
    plt.legend()
    plt.pause(0.1)

plt.show()
