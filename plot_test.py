import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
import numpy as np


y = np.array([0,1,2,3,4,5])
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })



fig, ax = plt.subplots(1, 2, figsize=(5.5,2.5))

# ax.plot(y)
# ax.plot(y*0.9)
# ax.plot(y*0.8)
# ax.plot(y*0.7)
# ax.set_xlabel('x label')
# ax.set_ylabel('y label')

# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
# Put a legend below current axis
# ax.legend(['only pose', 'pose + 20 lidar beams', 'only lidar, 20 beams', '4'], loc='upper center', bbox_to_anchor=(0.5, -0.3),
#           fancybox=True, shadow=False, ncol=2)

ax[0].plot(y)
ax[0].plot(y*0.9)
ax[0].plot(y*0.8)
ax[0].plot(y*0.7)
ax[0].set_xlabel('x label')
ax[0].set_ylabel('y label')

ax[1].plot(y)
ax[1].set_xlabel('x label')
ax[1].set_ylabel('y label')

#box = ax[0].get_position()
#ax[0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# Put a legend below current axis
#ax[0].legend(['only pose', 'pose + 20 lidar beams', 'only lidar, 20 beams', '4'], loc='upper center', bbox_to_anchor=(0.5, -0.3),
#          fancybox=True, shadow=False, ncol=2)

#ax[0].legend(['0', '1', '2'], bbox_to_anchor=(1.5, -0.5), loc='upper center', ncol=3)


fig.tight_layout()
fig.subplots_adjust(bottom=0.35) 


plt.figlegend(['0', '1', '2', '3'], loc = 'lower center', ncol=4, labelspacing=0.)
plt.savefig('test_wide.pgf', format='pgf')
#plt.show()