# Image Caption Generation README

This program generates captions for images using a deep learning model trained on a dataset of images and their corresponding captions. The model combines image features extracted using the VGG16 model and text features extracted using an LSTM-based architecture.

## Prerequisites

To run this program, you need to have the following dependencies installed:

- Python 3.x
- TensorFlow 2.x
- NumPy
- tqdm
- PIL (Python Imaging Library)

## Getting Started

1. Clone or download the program files to your local machine.

2. Download the pre-trained VGG16 model weights from the Keras library or any compatible source.

3. Ensure that you have a dataset of images and their corresponding captions in a suitable format. The images should be stored in a directory, and the captions should be in a separate text file.

4. Modify the `BASE_DIR` variable in the code to specify the base directory where your dataset and other files are located.

5. Run the program using a Python interpreter.

   ```
   python image_caption_2.py
   ```

## Usage

1. The program will load the VGG16 model and extract image features from the images in your dataset.

2. The captions data will be loaded and preprocessed, including cleaning and tokenization.

3. The model architecture will be created, consisting of an encoder (image feature layers) and a decoder (sequence feature layers).

4. The model will be trained using the training dataset, and the progress will be displayed for each epoch.

5. After training, the model will generate captions for images in the test dataset.

6. The generated captions will be compared with the actual captions, and the BLEU-1 and BLEU-2 scores will be calculated.

7. You can use the `generate_caption(image_name)` function to generate captions for specific images by providing their filenames.

## Notes

- Ensure that you have a suitable dataset of images and captions before running the program. The dataset should be organized with images in one directory and captions in a separate file.

- The VGG16 model weights should be downloaded separately and placed in the appropriate location. Update the `BASE_DIR` variable accordingly.

- Adjustments to the model architecture, training parameters, or other settings can be made in the code based on your requirements.

- The program uses the TensorFlow library for deep learning tasks. Make sure you have TensorFlow installed and compatible with your Python environment.

## License

This program is released under the [MIT License](LICENSE). Feel free to modify and distribute it according to your needs.

## Acknowledgments

- This program utilizes the VGG16 model and concepts from deep learning to generate captions for images.

- The TensorFlow library and other dependencies used in this program are essential tools for deep learning and natural language processing tasks.

- Special thanks to the developers and contributors of the mentioned libraries and tools for making this program possible.



generate models by running image caption 2.py , with location edited.
flickr 8k dataset in used.

Step1.: Make sure all the files are in same folder or directory.
Step2.: Open GUI.py.
Step3.: run  the code.
Step4.: Click on upload an image button.
        I have worked on small dataset around that have around 8000 .
        So, only the objects that are present in the image dataset 
        (like humans, dogs , kids ,ground, etc.) can be calsified. 
Step5.: Selected the image File.
Step6.: click on Generate Caption button.
Step7.: you have the required output.
