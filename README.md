# Insurance_Premium_Prediction

### Software And Tools Requirement

1. [Github Accounts](https://github.com)

2. [Azure account](https://azure.microsoft.com/en-in/)

3. [VSCodeIDE](https://code.visualstudio.com/Download)

4. [GitCLI](https://cli.github.com/)

Create a new enviornment for project
    
    
    conda create -p venv python==3.8 -y
    

Activate the enviornment
    
    conda activate venv/
    
Update Requirement libraries

    pip install -r requirements.txt

Create a pythonic file in which web application is made using flask

    app.py

Build the machine learning model using jupyter notebook
     
    file_name.ipynb

Get the output as a pickled file out of the built model

    regmodel.pkl and scaling.pkl

Create a templates folder to write all the html code

    index.html
    result.html
    form.html


For end to end project
     
    create folder src in which two more folder should be present components and pipeline
    In components data injestion , data transformation and model training is done
    In pipeline the are two pythonic scripts i.e train pipeline and predict pipeline is present
    Train Pipeline is responsible for training of the dataset
    Predict Pipeline is Responsible for prediction of any new data point from user using a flask app
    There are 3 more  pythonic file are present in src
    The first is the utils.py in which the common functionality function like save object ,load object function is written
    The second is the exception.py where custom exception code is written
    The third is the logger.py where code for logging of various activity is written




Finally add all the file to github

    git add .
    git status
    git commit -m "comments"
    git push origin main

Deployment

    For deployment create a account in azure
    Create a resource group
    create a webapp using github actions


