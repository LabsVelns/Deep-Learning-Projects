### Video Super Resolution

This fine-tuning algorithm imports an existing super-resolution algorithm (ESRGAN model) and fine-tunes it by producing a super-resolution (x4) image for each frame of the video. The perceptual loss between the super-resolution and the ground truth frame is calculated and propagated back through the network via gradient descent and the Adam optimizer.

### Major contributions

The significant contributions of this revision are:

1. Building a pipeline and adding features to generate super-resolution videos by processing one frame of a video at a time.
2. Building transformation tools to randomly select video frames and randomly cropped locations to avoid overfitting and speed up the training process.
3. Enabling gradients on the model.
4. Initializing an Adam optimizer (optimizer not provided in source code, nor mentioned in the paper).
5. Creating a feature extractor from VGG19 (mentioned in the ESRGAN paper but not provided in the code).
6. Using the feature extractor to create a perceptual loss.
7. Iterating over a training dataset and optimizing the model to reduce the perceptual loss.
8. Testing the model by generating super-resolution images using an unseen dataset (see `test_video_finetune.py`).


This repository contains a Streamlit web app that allows users to upload videos and process them with the ESRGAN video super-resolution model. The app is built using Python, Streamlit, and the ESRGAN model.

## How to use the app

To use the app, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Start the app by running `streamlit run deploy.py`.
4. Upload a video to the app.
5. Click the "Submit" button to process the video.
6. The processed video will be downloaded to your computer.

## Code explanation

The code for the app is divided into three files:

* `Architecture.py`: This file contains the definition of the ESRGAN model.
* `Blocks.py`: This file contains the definition of the building blocks used in the ESRGAN model.
* `deploy.py`: This file contains the code for the Streamlit app.
* `SidebySide.py`: This file contains the code that combines input and output video and presents on Streamlit.
* `Test & Train`: These files contain the code responsible for the training and testing of our model, one of which (`test.py`) is called in `deploy.py`

The `deploy.py` file is the entry point of the app. It imports the necessary libraries, defines the layout of the app, and handles the user input.

The `Architecture.py` file contains the definition of the ESRGAN model. The model is a convolutional neural network (CNN) that takes a low-resolution video frame as input and produces a high-resolution video frame as output. The model is based on the Residual Dense Block (RDB) architecture, a type of CNN specifically designed for image super-resolution.

The `Blocks.py` file contains the definition of the building blocks used in the ESRGAN model. These building blocks include convolutional layers, pooling layers, and activation functions.

## Conclusion

This app demonstrates how to use the ESRGAN model to perform video super-resolution. The app is easy to use and can be used to process videos of any size or resolution.

#                   `Output`------------------------------------------------------`Input`

<img width="1280" alt="image" src="https://github.com/LabsVelns/Deep-Learning-Projects/assets/68092358/963894d0-3264-48ab-9d1e-b748e9dbf10f">

#                   `Input`------------------------------------------------------`Output`

https://github.com/LabsVelns/Deep-Learning-Projects/assets/68092358/27cf9aea-45f2-471e-80e7-86c8426565aa

