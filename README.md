# Master_project

Code for an NLP project for my master thesis about 'Neural Intent Classifier for Open-Domain Dialog Systems by using
Multi-Task Learning' in Heriot-Watt university. 


This project focuses on using deep learning and multi-task learning techniques to build four neural intent classifier models for intent classification and parameter recognition tasks (the major tasks in NLU).

1. The intent model executes the intent classification task.
2. The param model executes the parameter recognition task.
3. The pipeline model combines the intent model and param model as a pipeline.
4. The hard parameter multi-task model executes both intent classification and parameter recognition tasks at the same time using the hard parameter sharing multi-task learning method.


The Figure below shows three deep learning model variants ("Intent Model", "Param Model", and "Hard Parameter Multi-Task Model") for intent classification and parameter recognition tasks.

<p align="center"><img src="https://user-images.githubusercontent.com/35661072/63275506-86de2100-c299-11e9-9774-c88a786854b1.png"></p>

This project builds and implements experiments of different models for the intent classification and parameter recognition tasks based on AllenNLP library.

Unfortunately, according to the confidential conditions of this project, the datasets used are not provided. An outputs example with intent and param label (BIO tagging scheme) shown below.

<p align="center"><img width="350" alt="An outputs example with intent and param label (BIO tagging scheme)" src="https://user-images.githubusercontent.com/35661072/65373327-e7cf8f00-dc73-11e9-9d58-8b640379d2af.png"></p>

### Result:
Obtained 80.81% F1-Score on test set by performing 7.15% better than baseline model.
