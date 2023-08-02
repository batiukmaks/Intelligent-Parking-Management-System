# Intelligent-Parking-Management-System

## Description
The purpose of this project is to develop an intelligent parking management system that will contribute to efficient traffic flow, enhanced security, and improved user experience within parking facilities.

## Project setup
Python version: 3.9.17

## Installation
- In the terminal run the command `git clone git@github.com:batiukmaks/Intelligent-Parking-Management-System.git`
- Then `pip install requirements.txt`
- Change the `config.yaml` to have your paths to dataset

## Training
- Change the `train_model.py` file: set the number of epochs, device, worker etc
- Run the command `python train_model.py`
In the result, you will have the weights for your model and examples of how the model works now. The files will be stored in your local storage, the exact path will be shown in the terminal.

## Testing
- Change the `test_model.py` file: change the path to the weights and the video to test the model on
- Run the command `python test_model.py`
In the result, the program will show the video with objects detected by your model.