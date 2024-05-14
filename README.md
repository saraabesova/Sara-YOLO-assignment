# Object Detection, Instance Segmentation and Mask Generation Task
<br />

## _Approach_

The yolo.py file contains code that performs three tasks: object detection, instance segmentation and mask generation. It uses the YOLOv8n model that was trained on the COCO dataset. The program detects specified objects in an input image and saves it as a JPG image. From here, it generates masks for each of the detected objects, and saves the resulting mask images as PNG files.

Before the actual generation of masks, I used the YOLOv8n model to identify the specified objects (put bounding boxes around them along with their probability score) and highlighting their pixels. This way, each relevant pixel is classified to its designated class. To generate the masks, I decided to use the image resulting from the instance segmentation. This way, the mask achieves precision by analyzing each pixel individually. To make the mask images black and white, I multiplied each pixel by 255. Each function has its description in the code.


<br />

## _Setup Instructions_

Running Through the Terminal
1. Download the folder with the files. Ensure all necessary files are included: yolo.py, image_14.png, requirements.txt.

2. Open the terminal/command prompt on your device. You can also use an integrated terminal in an python IDE, such Visual Studio Code or Pycharm.

3. Navigate to the folder containing the files mentioned in step 1.
   ```
   cd path/to/directory
   ```
   <br />
   Example:
   
   ```
   cd Documents/neolocus_assignment
   ```

5. Create a virtual environment.
   ```
   python -m venv venv
   ```
   <br />
   Example:
   
   ```
   python -m venv yolo_venv
   ```
6. Activate the virtual environment.
   <br />
   Example:
   ```
   cd yolo_venv
   ```
   ```
   source bin/activate
   ```
   
7. Run the requirements.txt file to install all the required packages. 
   ```
   pip install -r /path/to/requirements.txt
   ```
   <br />
   Example:
   
   ```
   pip install -r /Users/saraabesova/Documents/neolocus_assignment/requirements.txt
   ```
   <br />
   After the requirements are installed, it is time to run the script.
   <br />
   
## Usage Examples
8. Run the script with the following command:
    ```
    python <path_to_script> --image-path <path_to_image>
    ```
    Example:
    ```  
    python /Users/saraabesova/Documents/neolocus_assignment/yolo.py --image-path /Users/saraabesova/Documents/neolocus_assignment/image_14.jpg

    ```
    
After that, the masks will be generated and saved to your local folder. You should be able view them there!

### Object Detection Image
![image0](https://github.com/saraabesova/Sara-YOLO-assignment/assets/119079376/9e5093eb-dd7c-410a-b48d-6e3037f2c491)
<br />

### Instance Segmentation Image
![output_image](https://github.com/saraabesova/Sara-YOLO-assignment/assets/119079376/f62509d4-db31-4d40-8c2f-d218bd0bd534)
<br />

### An Example of a Mask
![mask_example](https://github.com/saraabesova/Sara-YOLO-assignment/assets/119079376/f09a80f6-0b82-4216-9cdb-738b9c1d0df6)




