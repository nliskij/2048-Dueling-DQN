import csv
import matplotlib.pyplot as plt
import numpy as np

csvfile = open("log.csv", "r")
log = csv.reader(csvfile)

rows = [row for row in log]

scores = [int(row[0]) for row in rows]
maxs = [np.log2(int(row[1])) for row in rows]
moves = [int(row[2]) for row in rows]
print(maxs.count(11))
print(len(maxs))
plt.subplot(1, 3, 1)
plt.hist(scores)
plt.subplot(1, 3, 2)
plt.hist(maxs, bins=np.arange(min(maxs), max(maxs) + 2))
plt.subplot(1, 3, 3)
plt.hist(moves)

plt.show()
