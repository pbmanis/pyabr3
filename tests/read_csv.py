import matplotlib.pyplot as mpl
import pandas as pd
from pathlib import Path
import scipy
import numpy as np

fn = "E:/Users/experimenters/Desktop/ABR_Code/Noise_Measure_2026.02.26_10.44.01.csv"
assert Path(fn).is_file()

d = pd.read_csv(fn)
print(d.columns)

fs = 1./np.mean(np.diff(d['Time']))
print("Fs = ", fs)
freqs, psd = scipy.signal.welch(d['Voltage(V)'], fs=fs)
f, ax = mpl.subplots(2, 1)
ax[0].plot(d['Time'], d['Voltage(V)'])
ax[1].plot(freqs, psd)
ax[1].set_xscale('log')
ax[1].set_yscale('log')
mpl.show()
