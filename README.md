# CoordiMotionApp

CoordiMotionApp is a Python application designed to evaluate interjoint coordination using advanced computational metrics. It leverages the [Coordination Metrics Toolbox](https://github.com/oceanedbs/CoordinationMetricsToolbox) and implements methods described in the [PLOS ONE paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0325792).

## Features

- Analyze and quantify interjoint coordination from videos
- Utilizes state-of-the-art algorithms from the Coordination Metrics Toolbox
- User-friendly interface for loading and processing data

## Dependencies

- [Python](https://www.python.org/)
- [mediapipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
- [scikit-learn](https://scikit-learn.org/)
- [scipy](https://scipy.org/)

## Installation & Build

To build the application, use:

```bash
xvfb-run -a pyinstaller Coordimotion.spec
```

## Usage

After building, run the app with:

```bash
./dist/Coordimotion
```

You can download the executable [here](https://cloud.isir.upmc.fr/s/GNb84QkZ3acRTz2)

## References

- [Coordination Metrics Toolbox](https://github.com/oceanedbs/CoordinationMetricsToolbox)
- [PLOS ONE Article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0325792)