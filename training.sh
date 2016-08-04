#!/bin/bash

cd scripts

# First stage
python model1.py
python model2.py
python model3.py
python model4.py
python model5.py

python ensemble1.py

# Second stage
python model6.py
python model7.py
python model8.py

python ensemble2.py

# Third stage
python model9.py
python model10.py

python ensemble3.py

cd ..
