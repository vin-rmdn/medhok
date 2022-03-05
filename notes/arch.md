# Architecture notes


VoxCeleb (arch referred to by Shon et al.):
## =============== Feature input ===============
- Mono
- 16-bit
- 16kHz for consistency

### Framing
- **25ms** hamming window (Nagrani et al., 2017; Warohma et al., 2018; Snyder et al., 2017)
    - 8000 samples / 1000ms * 25ms = 200 samples
        - Samples = Sample / second
        - Second = 1000ms

### Window
- 3 seconds (Snyder et al., 2017)

### Step size
- **10ms** step (Nagrani et al., 2017)
- 60% overlapping window (Warohma et al., 2018)
    - 25 ms * 0.6 = **15ms** step size

### Hamming Window
- To minimize the effect of signal discontinuity at each beginning and end of the frame
- Looks like Gaussian distribution
- Formula: W(n) = 0.54 * 0.46 cos(2πn / N-1)
> Used by Nagrani et al. (2017) and Warohma et al. (2018)

## Feature Preprocessing
- CMVN (Shon et al., 2018; Snyder et al., 2017)
    - Over a sliding window up to 3 seconds
- VAD (Shon et al., 2018; Snyder et al., 2017)
    -

## Features
MFCC: (Warohma et al., 2018; Snyder et al., 2017)
- Methods of obtaining:
    - STFT -> Mel Frequency Banks -> DCT (type-2) (Warohma et al., 2018)
- alternatively available as librosa.feature.mfcc

- Magnitude Spectrogram: 25ms hamming window, 10ms step; size: 512x300 for 3 seconds
- Mean and Variance normalisation is crucial for almost 10% increase in classification accuracy.
- Not using pre-processing features such as VAD, silence removal, or removal of unvoiced speech
- 3 second used by VoxCeleb (Nagrani et al., 2017)
[Arch]
- Basing on VGG-M (Return of the Devil in the Details: Delving Deep into Convolutional Nets by K. Chatfield et al.)
- (Warohma et al., 2018) Framing: 25ms, 16kHz.
    > 25ms means 40 data per second. Below 20ms (50 data per second), "it may cause each frame to lose its spectral estimated reliability and its differentiating power may decrease." (Warohma et al., 2018)
    > Currently we use 30 data per second (33.33ms)

[Feature types]
MFCC:
- 13 coefficients (Warohma et al., 2018)
- 20 coefficients (Snyder et al., 2017)

CNN-M (referred to as VGG-M by VoxCeleb):
model = tf.keras.Sequential([
            # conv1
            tf.keras.layers.Conv2D(96, (7, 7), 2, input_shape=_shape),
            tf.keras.layers.MaxPooling2D(2, 2),
            # conv2
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(256, (5, 5), 2),
            tf.keras.layers.MaxPooling2D(2, 2),
            # conv3
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(512, (3, 3), 1),
            # conv4
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(512, (3, 3), 1),
            # conv5
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(512, (3, 3), 1),
            tf.keras.layers.MaxPooling2D(2, 2),
            # full6
            tf.keras.layers.Dense(4096, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            # full7
            tf.keras.layers.Dense(4096, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            # full8
            tf.keras.layers.Dense(16, activation=tf.nn.softmax)
        ])

## Model
Time-delay neural network (Snyder et al., 2017)


## Loss
Categorical Cross-entropy (Snyder et al., 2017)
