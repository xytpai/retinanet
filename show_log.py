import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import scipy.signal as signal

e = np.load('log_eval_loss.npy')
t = np.load('log_train_loss.npy')
e = signal.medfilt(e,5)
t = signal.medfilt(t,5)
plt.plot(t, color='b', label='train')
plt.plot(e, color='r', label='eval')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode='expand', borderaxespad=0.)
plt.show()
