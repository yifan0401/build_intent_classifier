# Master_project

Code for an NLP project for my master thesis about 'Neural Intent Classifier for Open-Domain Dialog Systems by using
Multi-Task Learning' in Heriot-Watt university. 


This project focuses on using deep learning and multi-task learning techniques to build four neural intent classifier models for intent classification and parameter recognition tasks (the major tasks in NLU).

1. The intent model executes the intent classification task.
2. The param model executes the parameter recognition task.
3. The pipeline model combines the intent model and param model as a pipeline.
4. The hard parameter multi-task model executes both intent classification and parameter recognition tasks at the same time using the hard parameter sharing multi-task learning method.


The Figure below shows three deep learning model variants ("Intent Model", "Param Model", and "Hard Parameter Multi-Task Model") for intent classification and parameter recognition tasks.

![Introduction_Model](https://user-images.githubusercontent.com/35661072/63275506-86de2100-c299-11e9-9774-c88a786854b1.png)


This project builds and implements experiments of different models for the intent classification and parameter recognition tasks based on AllenNLP library.


Unfortunately, according to the confidential conditions of this project, the datasets used are not provided.
