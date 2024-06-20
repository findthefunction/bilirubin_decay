import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Data points
hours = np.array([13, 24, 36])
bilirubin_levels = np.array([147, 213, 170])

# Use a linear interpolation for simplicity
df = pd.DataFrame({'hours': hours, 'bilirubin_levels': bilirubin_levels})
df.set_index('hours', inplace=True)

# Resample to interpolate values
projected_hours = np.arange(0, 85, 1)
projected_levels = np.interp(projected_hours, df.index, df['bilirubin_levels'])

# Apply a hypothetical decay function for light treatment effect
decay_rate = 0.05  # assuming 5% decrease per hour due to treatment
for i in range(len(projected_levels)):
    if projected_hours[i] > 36:  # assuming treatment starts after 36 hours
        projected_levels[i] = projected_levels[i - 1] * (1 - decay_rate)

# Plotting the data and the model
plt.figure(figsize=(10, 6))
plt.plot(hours, bilirubin_levels, 'ro', label='Actual Bilirubin Levels')
plt.plot(projected_hours, projected_levels, 'b-', label='Projected Bilirubin Levels (with light treatment)')
plt.xlabel('Time (hours)')
plt.ylabel('Bilirubin Level (μmol/L)')
plt.title('Projected Bilirubin Levels with Light Treatment')
plt.legend()
plt.grid(True)
plt.show()
