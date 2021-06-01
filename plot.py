import matplotlib.pyplot as plt
import numpy as np

f = open("output.txt", 'r')
lines = f.readlines()
# plt.locator_params(axis='y', nbins=2)
# plt.locator_params(axis='x', nbins=2)
axes = plt.axes()

mini = int(lines[0].split(',')[0])
maxi = int(lines[0].split(',')[0])
x = []
y = []

for line in lines:
    x1, y1 = line.split(',')
    x.append(int(x1)) 
    y.append(int(y1))
    if int(x1) < mini:
        mini = int(x1)
    if int(x1) > maxi:
        maxi = int(x1)

plt.plot(x, y, 'ro')

print(mini, maxi)

# Plot y=x line
temp = np.arange(mini, maxi, 0.1)
plt.plot(temp, temp)

# Set axis labels
plt.xlabel("True Atomization Energy (kcal/mol)")
plt.ylabel("Predicted Atomization Energy (kcal/mol)")

# Show only 3 values on the axes:
# x_values = axes.get_xticks()
# y_values = axes.get_yticks()

# x_len = len(x_values)
# y_len = len(y_values)
# print(x_len)
# print(y_len)

# new_x = [x_values[i] for i in [0, x_len // 2, -1]]
# new_y = [y_values[i] for i in [0, y_len // 2, -1]]

# axes.set_xticks(new_x)
# axes.set_yticks(new_y)


plt.show()
f.close()
