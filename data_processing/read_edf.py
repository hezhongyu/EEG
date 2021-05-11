# -*- coding: UTF-8 -*

import mne
import matplotlib.pyplot as plt


data_file = 'd:/data/eeg/测试数据/癫痫/3173.edf'

raw = mne.io.read_raw_edf(data_file,
                    eog=None,
                    misc=None,
                    stim_channel='auto',
                    exclude=(),
                    preload=False,
                    verbose=None)
print(raw.info)
events_from_annot, event_dict = mne.events_from_annotations(raw)
print(event_dict)
print(events_from_annot)

data, times = raw[1:2, :]
plt.plot(times, data.T)
plt.title("Sample channels")
plt.show()
