# William Bouchard - 260866425
# Michel-Alexandre Riendeau - 260868849

import argparse
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from enum import Enum
import os
from scipy import sparse
import time
import statistics


class Type(Enum):
    Naive = 0
    FFT = 1


class Mode(Enum):
    DFT = 0
    Inverse = 1


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', help='mode', required=False,
                            type=int, action='store', dest='mode', default=1)
        parser.add_argument('-i', help='image', required=False,
                            type=str, action='store', dest='image', default='moonlanding.png')
        parsed_args = parser.parse_args()
    except:
        print('ERROR\tIncorrect input syntax. Check the arguments passed and try again')
        return

    mode = parsed_args.mode
    image = parsed_args.image
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    width = len(img[0])
    height = len(img)

    # Resize image to a power of 2 if not already a power of 2
    if width != 2 ** int(np.log2(width)):
        width = 2 ** (int(np.log2(width)) + 1)
    if height != 2 ** int(np.log2(height)):
        height = 2 ** (int(np.log2(height)) + 1)
    img = cv2.resize(img, (width, height))

    if mode == 1:
        naive_dft(img)
        fast_mode(img)
    elif mode == 2:
        denoising_mode(img)
    elif mode == 3:
        compressing_mode(img)
    elif mode == 4:
        plotting_runtime_mode()
    else:
        print('ERROR\tMode should be between 1 and 4')
        return


def fast_mode(img):
    fft_form = dft_or_inverse_2d(img, Type.FFT, Mode.DFT)
    # fft_form = np.fft.fft2(img)

    # Display
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(np.abs(fft_form), norm=LogNorm())
    plt.colorbar()
    plt.title('Fourier Transform')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def denoising_mode(img):
    fft_form = dft_or_inverse_2d(img, Type.FFT, Mode.DFT)
    # Source: http://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html
    rows, cols = fft_form.shape

    plt.figure(figsize=(15, 15))
    plt.subplot(3, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])

    # Removing low frequency
    low_freqs = fft_form.copy()
    keep_ratio = 0.1
    low_freqs[:int(rows * keep_ratio), :int(cols * keep_ratio)] = 0
    low_freqs[:int(rows * keep_ratio), int(cols * (1 - keep_ratio)):] = 0
    low_freqs[int(rows * (1 - keep_ratio)):, :int(cols * keep_ratio)] = 0
    low_freqs[int(rows * (1 - keep_ratio)):, int(cols * (1 - keep_ratio)):] = 0
    count = np.count_nonzero(low_freqs)
    denoised_form = dft_or_inverse_2d(
        low_freqs, Type.FFT, Mode.Inverse).real
    print('The number of non-zero entries for removing low frequencies with keep ratio of {} is: {}'.format(keep_ratio, count))
    print('The fraction they represent is: {}%\n'.format(
          (count / (rows * cols)) * 100))

    plt.subplot(3, 3, 2), plt.imshow(denoised_form, cmap='gray')
    plt.title('Denoised (Low Freq, keep ratio={})'.format(keep_ratio))
    plt.xticks([])
    plt.yticks([])

    keep_fractions = [0.05, 0.10, 0.15, 0.20, 0.25]
    index = 3
    for keep_fraction in keep_fractions:
        low_bound_row = int(rows * keep_fraction)
        high_bound_row = int(rows * (1 - keep_fraction))
        low_bound_col = int(cols * keep_fraction)
        high_bound_col = int(cols * (1 - keep_fraction))

        # Removing high freqs
        high_freqs = fft_form.copy()
        high_freqs[low_bound_row:high_bound_row] = 0
        high_freqs[:, low_bound_col:high_bound_col] = 0
        count = np.count_nonzero(high_freqs)
        denoised_form = dft_or_inverse_2d(
            high_freqs, Type.FFT, Mode.Inverse).real
        print('The number of non-zero entries for removing high frequencies with keep ratio of {} is: {}'.format(keep_fraction, count))
        print('The fraction they represent is: {}%\n'.format(
            (count / (rows * cols)) * 100))

        plt.subplot(3, 3, index), plt.imshow(denoised_form, cmap='gray')
        plt.title(
            'Denoised version (High Freq, keep ratio={})'.format(keep_fraction))
        plt.xticks([])
        plt.yticks([])

        index += 1

    # Removing based on mean threshold
    mean_threshold = fft_form.copy()
    mean = np.average(abs(mean_threshold))
    mean_threshold[mean_threshold < mean] = 0
    count = np.count_nonzero(mean_threshold)
    denoised_form = dft_or_inverse_2d(
        mean_threshold, Type.FFT, Mode.Inverse).real
    print('The number of non-zero entries for thresholding based on mean is: {}'.format(count))
    print('The fraction they represent is: {}%\n'.format(
        (count / (rows * cols)) * 100))

    plt.subplot(3, 3, index), plt.imshow(denoised_form, cmap='gray')
    plt.title('Denoised version (Thresholding based on mean)')
    plt.xticks([])
    plt.yticks([])

    plt.show()


def compressing_mode(img):
    fft_form = dft_or_inverse_2d(img, Type.FFT, Mode.DFT)
    compressions = [19, 38, 57, 76, 95]

    sparse.save_npz('og.npz', sparse.csr_matrix(fft_form))
    original_size = os.path.getsize('og.npz')
    print('0%\tthe number of non-zero entries is: ' +
          str(np.count_nonzero(fft_form)))

    plt.figure(figsize=(15, 5))
    plt.subplot(231), plt.imshow(img, cmap='gray')
    plt.title('Original Image, size = ' + str(original_size) + ' bytes')
    plt.xticks([])
    plt.yticks([])

    index = 2
    for compression_factor in compressions:
        compressed = fft_form.copy()

        # Compressing
        # We take absolute value to only keep the coefficients' magnitude
        percentile = np.percentile(abs(compressed), compression_factor)
        rows, cols = compressed.shape
        for i in range(rows):
            for j in range(cols):
                if abs(compressed[i][j]) < percentile:
                    compressed[i][j] = 0

        print(str(compression_factor) + '%\tthe number of non-zero entries is: ' +
              str(np.count_nonzero(compressed)))
        sparse.save_npz(str(compression_factor) + '.npz',
                        sparse.csr_matrix(compressed))
        size = os.path.getsize(str(compression_factor) + '.npz')

        inverse = dft_or_inverse_2d(compressed, Type.FFT, Mode.Inverse).real
        plt.subplot(2, 3, index), plt.imshow(inverse, cmap='gray')
        plt.title('Compressed ' + str(compression_factor) +
                  '%, size = ' + str(size) + ' bytes')
        plt.xticks([])
        plt.yticks([])
        index += 1

    plt.show()


def plotting_runtime_mode():
    arrays = [
        np.random.rand(2 ** 5, 2 ** 5),
        np.random.rand(2 ** 6, 2 ** 6),
        np.random.rand(2 ** 7, 2 ** 7),
        np.random.rand(2 ** 8, 2 ** 8),
    ]

    x_axis = ['32 x 32', '64 x 64', '128 x 128',
              '256 x 256']
    y_axis_naive = []
    y_axis_fast = []
    y_axis_naive_std = []
    y_axis_fast_std = []

    plt.figure(figsize=(15, 5))
    plt.title('DFT Runtime Analysis')
    plt.xlabel('Problem Size')
    plt.ylabel('Mean Runtime (sec)')

    for array in arrays:
        naive_runtimes = []
        fast_runtimes = []
        print('====== {} x {} ======'.format(array.shape[0], array.shape[1]))
        for _ in range(10):
            # Naive Runtime
            t1 = time.time()
            dft_or_inverse_2d(array, Type.Naive, Mode.DFT)
            t2 = time.time()
            naive_runtimes.append(t2 - t1)

            # Fast Runtime
            t1 = time.time()
            dft_or_inverse_2d(array, Type.FFT, Mode.DFT)
            t2 = time.time()
            fast_runtimes.append(t2 - t1)

        # We get the means and standard deviations
        naive_mean = np.average(naive_runtimes)
        naive_std = statistics.stdev(naive_runtimes)
        fast_mean = np.average(fast_runtimes)
        fast_std = statistics.stdev(fast_runtimes)

        print('Naive DFT - mean: {}, standard deviation: {}'.format(naive_mean, naive_std))
        print('FFT - mean: {}, standard deviation: {}'.format(fast_mean, fast_std))

        y_axis_naive.append(naive_mean)
        y_axis_naive_std.append(naive_std * 2)
        y_axis_fast.append(fast_mean)
        y_axis_fast_std.append(fast_std * 2)

    # 97 % confidence interval (2 * std)
    plt.errorbar(x_axis, y_axis_naive,
                 yerr=y_axis_naive_std, label='Naive')
    plt.errorbar(x_axis, y_axis_fast,
                 yerr=y_axis_fast_std, label='Fast')
    plt.show()


def naive_dft(x):
    # Naive implementation of DFT in 1D

    # Source https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/

    x = np.asarray(x.copy(), dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    e = np.exp(-2j * np.pi * n.reshape((N, 1)) * n / N)
    X = e @ x
    return X


def naive_inverse_dft(X):
    # Naive implementation of inverse DFT in 1D

    X = np.asarray(X.copy(), dtype=complex)

    N = X.shape[0]
    n = np.arange(N)
    e = (1 / N) * np.exp(2j * np.pi * n.reshape((N, 1)) * n / N)
    x = e @ X
    return x


def fft(x):
    # Implementation of Cooley-Tuckey FFT in 1D

    # Source https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/

    x = np.asarray(x.copy(), dtype=complex)
    N = x.shape[0]

    if N <= 32:  # We stop when we reach an arbitrary size of 32
        return naive_dft(x)

    even = fft(x[::2])
    odd = fft(x[1::2])
    e = np.exp(-2j * np.pi * np.arange(N) / N)

    return np.concatenate([even + e[:int(N / 2)] * odd, even + e[int(N / 2):] * odd])


def inverse_fft(X, itr=0):
    # Implementation of inverse Cooley-Tuckey FFT in 1D

    X = np.asarray(X.copy(), dtype=complex)
    N = X.shape[0]

    if N <= 32:
        # Multiply by N to cancel the division by N done when getting the inverse
        return naive_inverse_dft(X) * N

    even = inverse_fft(X[::2], itr + 1)
    odd = inverse_fft(X[1::2], itr + 1)
    e = np.exp(2j * np.pi * np.arange(N) / N)

    result = np.concatenate(
        [even + e[:int(N / 2)] * odd, even + e[int(N / 2):] * odd])

    if itr == 0:
        return (1 / N) * result  # To make sure we only divide by N once
    return result


def dft_or_inverse_2d(x, implementation: Type, mode: Mode):
    x_copy = x.copy()
    x_transposed = x_copy.transpose()
    x_column_modified = np.asarray(x_transposed, dtype=complex)

    for i, col in enumerate(x_transposed):
        if mode == Mode.DFT:  # We run DFT
            if implementation == Type.Naive:
                x_column_modified[i] = naive_dft(col)
            elif implementation == Type.FFT:
                x_column_modified[i] = fft(col)

        elif mode == Mode.Inverse:  # We run inverse DFT
            if implementation == Type.Naive:
                x_column_modified[i] = naive_inverse_dft(col)
            elif implementation == Type.FFT:
                x_column_modified[i] = inverse_fft(col)

    x_modified = np.asarray(x, dtype=complex)

    for j, row in enumerate(x_column_modified.transpose()):
        if mode == Mode.DFT:  # We run DFT
            if implementation == Type.Naive:
                x_modified[j] = naive_dft(row)
            elif implementation == Type.FFT:
                x_modified[j] = fft(row)

        elif mode == Mode.Inverse:  # We run inverse DFT
            if implementation == Type.Naive:
                x_modified[j] = naive_inverse_dft(row)
            elif implementation == Type.FFT:
                x_modified[j] = inverse_fft(row)

    return x_modified


if __name__ == '__main__':
    main()
