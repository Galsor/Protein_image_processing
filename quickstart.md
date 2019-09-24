# Quickstart
##For Linux user:
###Pre-requisites:
```
sudo apt install python3
sudo apt install python-pip
sudo apt install python3-tk 
sudo pip install virtualenv
```
This will install python3 that we will use and the package virtualenv in order to create a specific environment for our project.
From now on, stay in the directory containing the project

###Project installation and launch
To install the project:
```
git clone https://github.com/Galsor/Protein_image_processing.git
```
To create an environment named "venv" using python3:
```
virtualenv venv --python=python3
```
To activate and be in your environment, be in the directory containing the environment (here named venv):
```
source venv/bin/activate
```
(If you want to deactivate from your environment):
```
deactivate
```
Install all the packages needed in the environment using the file requirements.txt:
```
pip install -r requirements.txt
```
To launch jupyter lab:
```
jupyter lab
```
Now a window on your browser will pop up in the directory of the project. You can open the notebook with the extension .ipynb. You can run the code on your notebook.

##For Mac OS user:
###Pre-requisites:

This will install python3 that we will use and the package virtualenv in order to create a specific environment for our project.
```
# Install Homebrew 
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# Install Python
brew install python

# Install pip
sudo easy_install pip
# Install virtualenv (if you want to work on virtual environnement, which is recommended if you work on several python projects)
sudo pip install virtualenv
```
###Project installation and launch
To install the project:
```
git clone https://github.com/Galsor/Protein_image_processing.git
```
To create an environment named "venv" using python3:
```
virtualenv venv --python=python3
```
To activate and be in your environment, be in the directory containing the environment (here named venv):
```
source venv/bin/activate
```
To launch jupyter lab:
```
jupyter lab 
```
Now a window on your browser will pop up in the directory of the project. You can open the notebook with the extension .ipynb. You can run the code on your notebook.

##For Windows user:
###Install Conda:

- Download Anaconda Python3.7 for Windows [https://www.anaconda.com/download/]
- Double-click the .exe file
- Follow the instructions

###Project installation and launch
Setup the virtual environment:

- Open Anaconda, go to /Environments
- Create a new environment "venv" with python3.6
- Click on play then "open terminal"

Go in the directory you want to use for the project:
```
cd C:\Users\...
```
Install the project and update packages with pip:
```
git clone https://github.com/Galsor/Protein_image_processing.git
pip install -r requirements.txt
```

To launch jupyter lab:
```
Go to Anaconda/Environments, select "venv" and launch with jupyter Notebook
```
Or, from the terminal:
```
activate venv
jupyter lab
```