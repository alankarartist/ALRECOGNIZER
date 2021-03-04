# AlRecognizer

A python script for face recognition using OpenCV.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install packages.

```bash
pip install numpy
pip install opencv_contrib_python
```

## Usage

![](/CaptureTrain.JPG)

![](/CaptureTest.JPG)

1. In folder ./AlRecognizer/TrainingImages/1 keep the set of images of one person and in ./AlRecognizer/TrainingImages/0 keep set of images of another person.
2. In ./AlRecognizer/TrainingImages folder keep the image you want to test.
3. Run AlTrainer.py, before running change line 6 for test image and line 18 to classify the faces. At last, test image will be shown with person's name as shown in first image.
4. Then, change line 9 in AlLiveRecognizer.py and run the script to identify the face on live feed, as shown in second image.
5. You can ignore or also delete AlRecognizer.py.