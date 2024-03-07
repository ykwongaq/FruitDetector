# Fruit Detector

This project is used to detect the 3D position of following fruit 🍎:

| Fruit ID  | Name |
|--------|--------|
| 0  | Banana    |
| 1 | Orange |
| 2 | Pomegranate |
| 3 | Grapes |
| 4 | Raspberries |

The output 3D position (xyz) represent the offset from the origin (position of camera) in term `mm`
## How to use

### 1. Set up Anaconda environment

#### Step 1: Install Anaconda

1. Download the Anaconda installer for Linux from the [official website](https://www.anaconda.com/products/distribution#download-section).
2. Open a terminal.
3. Navigate to the directory where you downloaded the installer.
4. Run the following command to install Anaconda: `bash Anaconda3-2021.05-Linux-x86_64.sh` (replace with the name of the file you downloaded).
5. Follow the prompts on the installer screens.
6. If you are unsure about any setting, accept the defaults. You can change them later.
7. To make the changes take effect, close and then re-open your terminal window.

#### Step 2: Create an Anaconda Environment

1. Open a terminal.
2. Navigate to the directory containing the `environment.yaml` file.
3. Run the following command to create an environment: `conda env create -f environment.yaml`.
4. Activate the new environment by running: `conda activate myenv` (replace `myenv` with the name of your environment, which is specified in the `environment.yaml` file).

Now you have set up your Anaconda environment and are ready to start using it!

### 2. Configura fruit detector

In `configs/locator.yaml`, please change the configuration `model.weight` to the path to the model. Here is the example

```
color_camera:
  fx: 912.782470703125
  fy: 912.799072265625
  cx: 641.11669921875
  cy: 370.3597106933594

depth_camera:
  fx: 636.6630249023438
  fy: 636.6630249023438
  cx: 647.2459716796875
  cy: 355.68048095703125

model:
  weight: /home/davidwong/documents/FruitDetector/outputs/yolom_20240306/weights/best.pt
  conf: 0.5
  height: 720
  width: 1280
```

### 3. Get 3D position

Read and try to run the code `demo.py`. You need to provide the input `.png` image and the `.npy` depth map.

Here is an example:

```
python demo.py --image_path <path to png image> --depth_path <path to .npy depth map>
```

Replace the corresponding input to the absolute path of required file

### Addition setup

If you want to show the result of object detection, you can execute the following command

```
python demo.py --image_path <path to png image> --depth_path <path to .npy depth map> --show_output
```

If you want to save the result as image, you can execute the following cammand
```
python demo.py --image_path <path to png image> --depth_path <path to .npy depth map> --save_output
```
It will save the output to the default folder `./location_results`


If you want to specify the location results folder, you can execute the following cammand
```
python demo.py --image_path <path to png image> --depth_path <path to .npy depth map> --save_output --output_folder <path to output folder>
```