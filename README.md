# Fine-tune-stable-diffusion-model-runwayml-stable-diffusion-v1-5-with-custom-Dataset-img-to-img-task

This project is a finetune project of a stable diffusion model (runwayml\stable-diffusion-v1-5) with a custom dummy dataset for the image to image task.

## Features
- **Finetune with custom small dummy fashion dataste:** Dataset with only 3 Photos and there description.
- **Image Generation:** The Finetuned model is in Google Drive, The link is given below, you can download it and change the path in code of img_to_img_test.py.

## Files and Folders
- `sample_imges`: This folder is for sample images which were used for train the model.
- `test_images`: This folder is for test the images with prompt
- `finetune.py`: This file used for finetuning
- `img_to_img_test.py`: This file used for test the model.
- `[Finetuned Model](https://drive.google.com/drive/folders/1R35fEcWAjjsgha5rFuFWk6v9w08x3DUg?usp=sharing)`: This is the finetuned model link. (I could not upload it in git because of it's size (3.20 GB).

## How to Run

### Step 1: Install Dependencies
Install all the required libraries by running the following command:
```bash
pip install -r requirements.txt
```

### Step 2: Run Finetune
Run the finetune.py directly in the terminal for train model again:
```bash
python finetune.py
```


### Step 3. Test the model
Test the model after finetuning by running the img_to_img_test file and you can also change the prompt and test image into the code
```bash
 python img_to_img_test.py
```

## License
This project is open-source and free to use. Feel free to modify it as per your needs.


