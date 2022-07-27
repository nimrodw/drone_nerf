import numpy as np
#using the pyproj library (thanks user2856!)

#lon, lat
ENU = np.array([35.0469230611111,32.6361373305556])
local_coords = np.array([235.8521412610665, -359.0777891504635, 372.6693892240654])

r_earth = 6371.00
new_latitude  = ENU[0]  + (local_coords[1] / r_earth) * (180.0 / np.pi)
new_longitude = ENU[1] + (local_coords[0] / r_earth) * (180.0 / np.pi) / np.cos(ENU[0] * np.pi/180.0)
print(new_latitude, ", ", new_longitude)
