import numpy as np
import matplotlib.pyplot as plt

def goertzel(freq, samples, sample_rate):
    """
    Calculate the Goertzel magnitude at a specific frequency.
    """
    k = int(0.5 + ((len(samples) * freq) / sample_rate))
    omega = (2.0 * np.pi * k) / len(samples)
    sine = np.sin(omega)
    cosine = np.cos(omega)
    coeff = 2.0 * cosine
    q0 = q1 = q2 = 0
    
    for sample in samples:
        q0 = coeff * q1 - q2 + sample
        q2 = q1
        q1 = q0
    
    real = (q1 - q2 * cosine)
    imag = (q2 * sine)
    magnitude = np.sqrt(real**2 + imag**2)
    return magnitude

def find_dominant_frequencies(samples, sample_rate, num_frequencies=5):
    """
    Find the top 'num_frequencies' dominant frequencies in the signal.
    """
    magnitudes = []
    frequencies = np.linspace(0, sample_rate/2, len(samples)//2)
    
    for freq in frequencies:
        magnitude = goertzel(freq, samples, sample_rate)
        magnitudes.append((freq, magnitude))
    
    # Sort based on magnitude
    magnitudes.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top 'num_frequencies' frequencies
    return magnitudes[:num_frequencies]

if __name__ == "__main__":
    fs = 8000  # Sample rate, Hz
    f = 400  # Frequency of the sine wave, Hz
    N = 2048  # Number of samples
    t = np.arange(N) / fs
    signal = 0.5 * np.sin(2 * np.pi * f * t) + np.random.normal(0, 0.05, N)
    #signal = np.sin(2 * np.pi * f * t)
    
    dominant_frequencies = find_dominant_frequencies(signal, fs, num_frequencies = 12)
    
    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    
    # Input signal
    ax[0].plot(t, signal)
    ax[0].set_title('Input Signal')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Amplitude')
    
    # Goertzel output (Dominant Frequencies)
    frequencies, magnitudes = zip(*dominant_frequencies)
    ax[1].bar(frequencies, magnitudes, width=fs/(2*N))
    ax[1].set_title('Top 5 Dominant Frequencies')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('Magnitude')

    plt.tight_layout()
    plt.show()