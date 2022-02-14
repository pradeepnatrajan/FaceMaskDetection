# FaceMaskDetection
![SystemDescription](https://user-images.githubusercontent.com/57305406/153955774-df8211e6-5bb5-42a7-831b-6337ab859492.png)
## Step by step guide for implement this project.
### Step 1:Import Required Libraries
```
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
```
### Step 2: Intialize learning rate and no. of epochs  for the model
```
INIT_LR= 1e-3 
EPOCHS = 20 
BS = 32
```


