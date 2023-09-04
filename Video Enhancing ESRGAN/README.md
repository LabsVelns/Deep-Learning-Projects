### Video Super Resolution

This fine-tuning algorithm imports an existing super-resolution algorithm (ESRGAN model) and fine-tunes it by producing a super-resolution (x4) image for each frame of the video. The perceptual loss between the super-resolution and the ground truth frame is calculated and propagated back through the network via gradient descent and the Adam optimizer.

### Major contributions

The major contributions of this revision are:

1. Building a pipeline and adding features to generate super-resolution videos by processing one frame of a video at a time.
2. Building transformation tools to randomly select video frames and random cropped locations to avoid overfitting and speed up the training process.
3. Enabling gradients on the model.
4. Initializing an Adam optimizer (optimizer not provided in source code, nor mentioned in the paper).
5. Creating a feature extractor from VGG19 (mentioned in the ESRGAN paper but not provided in the code).
6. Using the feature extractor to create a perceptual loss.
7. Iterating over a training dataset and optimizing the model to reduce the perceptual loss.
8. Testing the model by generating super-resolution images using an unseen dataset (see `test_video_finetune.py`).

