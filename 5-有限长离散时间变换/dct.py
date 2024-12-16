import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct, idct

np.set_printoptions(precision=1, suppress=True)

origin_pic = np.array(
    [
        [62, 55, 55, 54, 49, 48, 47, 55],
        [62, 57, 54, 52, 48, 47, 48, 53],
        [61, 60, 52, 49, 48, 47, 49, 54],
        [63, 61, 60, 60, 63, 65, 68, 65],
        [67, 67, 70, 74, 79, 85, 91, 92],
        [82, 95, 101, 106, 114, 115, 112, 117],
        [96, 111, 115, 119, 128, 128, 130, 127],
        [109, 121, 127, 133, 139, 141, 140, 133],
    ]
)
plt.figure("origin picture")
plt.imshow(origin_pic, cmap="gray")
plt.show()

centered_pic = origin_pic - 128

dct_pic = dct(centered_pic, 2, norm="ortho")

table = np.array(
    [
        [6, 12, 14, 14, 18, 24, 49, 72],
        [11, 12, 13, 17, 22, 35, 64, 92],
        [10, 14, 16, 22, 37, 55, 78, 95],
        [16, 19, 24, 29, 56, 64, 87, 98],
        [24, 26, 40, 51, 68, 81, 103, 112],
        [40, 58, 57, 87, 109, 104, 121, 100],
        [48, 60, 69, 80, 103, 113, 120, 103],
        [51, 55, 56, 62, 77, 92, 101, 99],
    ]
)
compressed_pic = np.around(dct_pic / table)
print(compressed_pic)

decode_pic = idct(compressed_pic * table, 2, norm="ortho") + 128
plt.figure("compressed picture")
plt.imshow(decode_pic, cmap="gray")
plt.show()