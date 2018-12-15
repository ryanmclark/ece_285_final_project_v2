# ece_285_final_project_v2

Description
===========
This is project, Image Segmentation of Aerial Images, is an extension of the PhD Thesis written by Mnih Volodymyr of the University of Toronto in 2013. In this project we utilize VGG-16 and ResNet- 18&34 layer architectures to preform semantic segmentation on Satellite imagery of the Greater Area of Boston in order to isolate and identify buildings and roads against the background.

Team:
-Ryan Marshall
-Taiwei Lu

Project Originator: Mnih Volodymyr
Link to PhD Thesis: https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf
Link to GitHub: https://github.com/mitmul/ssai-cnn
Link to Dataset: https://www.cs.toronto.edu/~vmnih/data/

Requirements
============
There are no additional packages that need to be installed that are not already native to the pod.
The packages that were used that are not native to python as are used as followed:

	numpy
	scipy
	matplotlib
	cv2
	torch
	torchvision
	tensorflow

Custom classes that must be a direct subfile in the ResNet directory:
	data_handler.py
	resnet.py

Code organization
=================
ResNet Layout:
    ResNet/
	test_buildings.ipynb (run to test and visualize output - necessary data is present)
	train_buildings.ipynb (trains building  - will not work without the data)
	buildings_model.pkl (The Trained Model for building detection)
	resnet.py (class that defines the architecture)
	data_handler.py (class that loads the data into batches)
	data/ (ignore contents - used in test_buildings.ipynb)
	utils/ (ignore contents - helpful for creation)

Directions: The only thing that needs to be ran is test_buildings.ipynb. The rest is there for viewing the content.

VGG-16 and TensorFlow Directions:
-Image segmentation using VGG16.ipynb -- Run code for training/visualization (download the support files in the folder 'VGG-SegNetwork'.To see the results, scroll down to demo execution on image at the bottom of the notebook) 

  A link to download the folder is:https://drive.google.com/drive/folders/1KqzNLe876O0xh99PvDBwHHemM2rxwsol?usp=sharing
