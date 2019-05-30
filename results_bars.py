#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def norm(x, alfa = 0):
    return (x - np.min(x)) / (np.max(x) - np.min(x)) + alfa

#prosecno zadovoljeno klauza po jednoj iteraciji
num_sat = np.array([407.0, 407.40, 436.35, 435.07, 435.30, 407.75, 438.69, 438.70])
#prosecno vreme jedne iteracije
time = np.array([0.01, 1.11, 8.64, 1.01, 0.77, 0.02, 9.18, 10.81])

norm_num_sat = norm(num_sat, 0.01)
norm_time = norm(time, 0.01)
result = norm(norm(num_sat,0.09)/norm(time,0.02), 0.01)

data = np.transpose(np.vstack((norm_num_sat, norm_time, result)))
#print(data)

length = len(data)
x_labels = ['SAWEA', 'RFEA', 'FlipGA', 'ASAP', 'MASAP', 'PSO-LS', 'PSOSAT', 'WPSOSAT']

# Set plot parameters
fig, ax = plt.subplots()
width = 0.2 # width of bar
x = np.arange(length)
plt.xticks(fontsize = 14)

ax.bar(x, data[:,0], width, color='#000080',
       label='prosek zadovoljenih klauza po iteraciji')
ax.bar(x + width, data[:,1], width, color='#0151B1',
       label='prosek trajanja iteracije')
ax.bar(x + (2 * width), data[:,2], width, color='#BB2115',
       label='FINALNI RANG: odnos zadovoljenih klauza i trajanja iteracije')

ax.set_ylim(0, 1.2)
ax.set_xticks(x + width + width/2)
ax.set_xticklabels(x_labels)
ax.set_xlabel('Algoritmi')
#ax.set_title('Title')
ax.legend()
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

fig.tight_layout()
plt.show()
