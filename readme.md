# **Skeleton-Guided Bodypart Segmentation for Thermal Imaging**

## **Overview**
This project integrates **DeepLabCut (DLC)** skeleton tracking with **Mask R-CNN** to improve object segmentation in thermal imaging. The goal is to use skeleton keypoints as guidance to enhance mask predictions, making it useful for tracking body regions in mice or other animals in **high-resolution thermal videos**.

---

## **Project Structure**
```
/sgbs
│── /data/                     # Folder for datasets
│── /src/                      # Source code folder
│    │── data/                 
│    │    ├── data_alignment.py         # Align segmentation and skeleton data
│    │    ├── dlc_skeleton_tracking.py  # DLC-based tracking
│    │── models/               
│    │    ├── sg_maskrcnn_train.py      # Model training script
│    │    ├── sg_maskrcnn_analysis.py   # Model evaluation and inference
│    │── utils/                
│    │    ├── utils.py                  # Helper functions
│    │    ├── visualize.py              # Visualization utilities
│── /results/                  # Stores outputs (predictions, model weights)
│── README.md                  # Project documentation
│── requirements.txt           # Required Python dependencies
```

---

## **Installation**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Skeleton-Guided-Segmentation.git
   cd Skeleton-Guided-Segmentation

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Mac/Linux
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure you have PyTorch installed with CUDA support (if using GPU):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

---

## **Usage**
### **1. Extract images, temperatures, and timestamps from `.seq` files**
Use `seq_process` for processing `.seq` files:
- See [seq_process GitHub](https://github.com/j-landen/seq_process) for more information.

### **2. Train & track keypoints with DeepLabCut**
- See [DeepLabCut GitHub](https://github.com/DeepLabCut/DeepLabCut) for more details.

### **3. Train the SGBS model**
Run the following command to train the Mask R-CNN model with skeleton guidance:
   ```bash
   python src/models/sg_maskrcnn_train.py
   ```

### **4. Analyze images**
Use the trained model to analyze images and extract keypoint-enhanced segmentations:
   ```bash
   python src/models/sg_maskrcnn_analysis.py
   ```

---

## **Key Features**
✅ **Multi-Modal Data Integration** – Uses both image and keypoint data to increase accuracy  
✅ **Custom Mask R-CNN Architecture** – Attention-based fusion layer for enhanced predictions  
✅ **DeepLabCut Integration** – Input keypoints from DeepLabCut  
✅ **Thermal Image Processing** – Supports segmentation on **Flir thermal camera data**  

---

## **Results**
- **Bounding Box and Mask Predictions:** Extracts object masks with improved accuracy using skeleton cues.  
- **Temperature Mapping:** Converts `.seq` thermal images to temperature-encoded segmentations.  
- **Visualization:** Uses `display_instances()` for overlaying keypoints and masks.  

## **Examples**
![alt text](https://github.com/j-landen/SGBS/blob/master/data/0.PNG?raw=true)  ![alt text](https://github.com/j-landen/SGBS/blob/master/results/0.png?raw=true) 
![alt text](https://github.com/j-landen/SGBS/blob/master/data/4.PNG?raw=true)  ![alt text](https://github.com/j-landen/SGBS/blob/master/results/4.png?raw=true) 

## **Confirmation of results**
![alt text](https://github.com/j-landen/SGBS/blob/master/results/comparison_epoch200_log.png?raw=true)
Comparison between base Mask R-CNN model & new Skeleton-guided bodypart segmentation on thermographic images.

---

## **Acknowledgments**
- **[DeepLabCut](https://deeplabcut.github.io/DeepLabCut/)** for skeleton tracking  
- **[PyTorch](https://pytorch.org/)** for model implementation  
- **[Matterport Mask R-CNN](https://github.com/matterport/Mask_RCNN)** for visualize package

If you use this project in your research, please cite it accordingly!  

---

## **License**
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## **Contributing**
We welcome contributions! Feel free to submit **pull requests** or open **issues**. If you have suggestions or improvements, join the discussion.
