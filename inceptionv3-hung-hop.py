#!/usr/bin/env python
# coding: utf-8

# ## [1. Imports](#im) ##
# ## [2. HyperParameters](#hp) ##
# ## [3. Data Loading and Preprocessing](#data) ##
# ## [4. InceptionV3 Model](#model)  ##
# ## [5. Training and Fine Tuning](#train) ##
# ## [6. Visualizing Results](#vis) ##

# <a id="im"></a>
# # <center>IMPORTING LIBRARIES</center> 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix , classification_report 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense 


import time

from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')


# <a id="hp"></a>
# # <center>HYPERPARAMETRERS AND DIRECTORIES</center>

# In[ ]:


# train_dir = "/kaggle/input/emotion-detection-fer/train"
train_dir = "/home/project-3/Sanosh/images/train"


# test_dir = "/kaggle/input/emotion-detection-fer/test"
test_dir = "/home/project-3/Sanosh/images/test"


SEED = 125 # giúp cho việc tái lặp các thí nghiệm có thể tái tạo được.
IMG_HEIGHT = 139 # 
# IMG_HEIGHT = 299 # 

IMG_WIDTH = 139
# IMG_WIDTH = 299

BATCH_SIZE = 64 # dữ liệu được sử dụng trong quá trình huấn luyện mô hình.
EPOCHS = 30 # vòng lặp qua toàn bộ dữ liệu huấn luyện
FINE_TUNING_EPOCHS = 20 # (tinh chỉnh) mô hình.
# FINE_TUNING_EPOCHS = 10 # (tinh chỉnh) mô hình.

LR = 0.01 #Tốc độ học
NUM_CLASSES = 7
EARLY_STOPPING_CRITERIA=3
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
CLASS_LABELS_EMOJIS = ["👿", "🤢" , "😱" , "😊" , "😐 ", "😔" , "😲" ]


# <a id="data"></a>
# # <center> DATA LOADING AND PRE-PROCESSING</center>

# In[ ]:


preprocess_fun = tf.keras.applications.inception_v3.preprocess_input

train_datagen = ImageDataGenerator(horizontal_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.05,
                                   rescale = 1./255,
                                   validation_split = 0.2,
                                   preprocessing_function=preprocess_fun
                                  )
test_datagen = ImageDataGenerator(rescale = 1./255,
                                  validation_split = 0.2,
                                  preprocessing_function=preprocess_fun)

train_generator = train_datagen.flow_from_directory(directory = train_dir,
                                                    target_size = (IMG_HEIGHT ,IMG_WIDTH),
                                                    batch_size = BATCH_SIZE,
                                                    shuffle  = True , 
                                                    color_mode = "rgb",
                                                    class_mode = "categorical",
                                                    subset = "training",
                                                    seed = 125
                                                   )

validation_generator = test_datagen.flow_from_directory(directory = train_dir,
                                                         target_size = (IMG_HEIGHT ,IMG_WIDTH),
                                                         batch_size = BATCH_SIZE,
                                                         shuffle  = True , 
                                                         color_mode = "rgb",
                                                         class_mode = "categorical",
                                                         subset = "validation",
                                                         seed = 125
                                                        )

test_generator = test_datagen.flow_from_directory(directory = test_dir,
                                                   target_size = (IMG_HEIGHT ,IMG_WIDTH),
                                                    batch_size = BATCH_SIZE,
                                                    shuffle  = False , 
                                                    color_mode = "rgb",
                                                    class_mode = "categorical",
                                                    seed = 125
                                                  )


# ## Images with different emotions

# In[ ]:


# Helper Functions
def display_one_image(image, title, subplot, color):
    plt.subplot(subplot)
    plt.axis('off')
    plt.imshow(image)
    plt.title(title, fontsize=16)
    
def display_nine_images(images, titles, title_colors=None):
    subplot = 331
    plt.figure(figsize=(13,13))
    for i in range(9):
        color = 'black' if title_colors is None else title_colors[i]
        display_one_image(images[i], titles[i], 331+i, color)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def image_title(label, prediction):
  # Both prediction (probabilities) and label (one-hot) are arrays with one item per class.
    class_idx = np.argmax(label, axis=-1)
    prediction_idx = np.argmax(prediction, axis=-1)
    if class_idx == prediction_idx:
        return f'{CLASS_LABELS[prediction_idx]} [correct]', 'black'
    else:
        return f'{CLASS_LABELS[prediction_idx]} [incorrect, should be {CLASS_LABELS[class_idx]}]', 'red'

def get_titles(images, labels, model):
    predictions = model.predict(images)
    titles, colors = [], []
    for label, prediction in zip(classes, predictions):
        title, color = image_title(label, prediction)
        titles.append(title)
        colors.append(color)
    return titles, colors

img_datagen = ImageDataGenerator(rescale = 1./255)
img_generator = img_datagen.flow_from_directory(directory = train_dir,
                                                   target_size = (IMG_HEIGHT ,IMG_WIDTH),
                                                    batch_size = BATCH_SIZE,
                                                    shuffle  = True , 
                                                    color_mode = "rgb",
                                                    class_mode = "categorical",
                                                    seed = 125
                                                  )
clear_output()

images, classes = next(img_generator)
class_idxs = np.argmax(classes, axis=-1) 
labels = [CLASS_LABELS[idx] for idx in class_idxs]
display_nine_images(images, labels)


# ## Data distribution (count) among differnt emotions

# In[ ]:


# fig = px.bar(x = CLASS_LABELS_EMOJIS,
#              y = [list(train_generator.classes).count(i) for i in np.unique(train_generator.classes)] , 
#              color = np.unique(train_generator.classes) ,
#              color_continuous_scale="Emrld") 
# fig.update_xaxes(title="Emotions")
# fig.update_yaxes(title = "Number of Images")
# fig.update_layout(showlegend = True,
#     title = {
#         'text': 'Train Data Distribution ',
#         'y':0.95,
#         'x':0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'})
# fig.show()


# <a id="model"></a>
# # <center> InceptionV3 Transfer Learning  </center>

# In[ ]:


def feature_extractor(inputs):
    feature_extractor = tf.keras.applications.InceptionV3(input_shape=(IMG_HEIGHT,IMG_WIDTH, 3),
                                               include_top=False,
                                               weights="imagenet")(inputs)
    
    return feature_extractor

def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    # x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    # x = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    # x = tf.keras.layers.Dropout(0.5) (x)
    x = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classification")(x)
    
    return x

def final_model(inputs):
    densenet_feature_extractor = feature_extractor(inputs)
    classification_output = classifier(densenet_feature_extractor)
    
    return classification_output

def define_compile_model():
    
    inputs = tf.keras.layers.Input(shape=(IMG_HEIGHT ,IMG_WIDTH,3))
    classification_output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)
     
    model.compile(optimizer=tf.keras.optimizers.SGD(0.1), 
                loss='categorical_crossentropy',
                metrics = ['accuracy'])
  
    return model


# ## Summary of model

# 

# In[ ]:


model = define_compile_model()
clear_output()

# Feezing the feature extraction layers
model.layers[1].trainable = False

model.summary()


# <a id="train"></a>
# # <center> Training and Fine-Tuning </center> 

# ## Training model with freezed layers of InceptionV3

# In[ ]:


earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                         patience=EARLY_STOPPING_CRITERIA,
                                                         verbose= 1 ,
                                                         restore_best_weights=True
                                                        )
starting_time = time.time()

history = model.fit(x = train_generator,
                    epochs = EPOCHS ,
                    validation_data = validation_generator , 
                    callbacks= [earlyStoppingCallback])

history = pd.DataFrame(history.history)
print('> training time is %.4f minutes' % ((time.time() - starting_time)/60))


# ## Fine Tuning

# In[ ]:


# Un-Freezing the feature extraction layers for fine tuning 
model.layers[1].trainable = True
starting_time = time.time()
model.compile(optimizer=tf.keras.optimizers.SGD(0.001), #lower learning rate
                loss='categorical_crossentropy',
                metrics = ['accuracy'])

history_ = model.fit(x = train_generator,
                     epochs = FINE_TUNING_EPOCHS ,
                     validation_data = validation_generator)
# history = history.concat(pd.DataFrame(history_.history) , ignore_index=True)
history = pd.concat([history, pd.DataFrame(history_.history)], ignore_index=True)
print('> training time is %.4f minutes' % ((time.time() - starting_time)/60))


# ## Training plots

# In[ ]:


x = px.line(data_frame= history , y= ["accuracy" , "val_accuracy"] ,markers = True )
x.update_xaxes(title="Number of Epochs")
x.update_yaxes(title = "Accuracy")
x.update_layout(showlegend = True,
    title = {
        'text': 'Accuracy vs Number of Epochs',
        'y':0.94,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
x.show()


# In[ ]:


x = px.line(data_frame= history , 
            y= ["loss" , "val_loss"] , markers = True )
x.update_xaxes(title="Number of Epochs")
x.update_yaxes(title = "Loss")
x.update_layout(showlegend = True,
    title = {
        'text': 'Loss vs Number of Epochs',
        'y':0.94,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
x.show()


# <a id="vis"></a>
# # <center> Visualizing Results </center> 

# ## Model Evaluation

# In[ ]:


model.evaluate(test_generator)
preds = model.predict(test_generator)
y_preds = np.argmax(preds , axis = 1 )
y_test = np.array(test_generator.labels)


# ## Confusion Matrix

# In[ ]:


cm_data = confusion_matrix(y_test , y_preds)
cm = pd.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (20,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')


# In[ ]:


# Tính phần trăm trong từng đối tượng của ma trận hỗn loạn
cm_percent = np.round(cm_data / cm_data.sum(axis=1)[:, np.newaxis] * 100, 2)

# Tạo DataFrame từ ma trận phần trăm
cm_percent_df = pd.DataFrame(cm_percent, columns=CLASS_LABELS, index=CLASS_LABELS)
cm_percent_df.index.name = 'Actual'
cm_percent_df.columns.name = 'Predicted'

# Vẽ heatmap của ma trận phần trăm
plt.figure(figsize=(20, 10))
plt.title('Confusion Matrix (Percentage)', fontsize=20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm_percent_df, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='.2f')

plt.show()


# ## Classification Report 

# In[ ]:


# 'Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"
print(classification_report(y_test, y_preds))


# ## Multiclass AUC Curve

# In[ ]:


fig, c_ax = plt.subplots(1,1, figsize = (15,8))

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    for (idx, c_label) in enumerate(CLASS_LABELS):
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr,lw=2, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'black',linestyle='dashed', lw=4, label = 'Random Guessing')
    return roc_auc_score(y_test, y_pred, average=average)

print('ROC AUC score:', multiclass_roc_auc_score(y_test , preds  , average = "micro"))
plt.xlabel('FALSE POSITIVE RATE', fontsize=18)
plt.ylabel('TRUE POSITIVE RATE', fontsize=16)
plt.legend(fontsize = 11.5)
plt.show()


# In[ ]:


print("ROC-AUC Score  = " ,roc_auc_score(to_categorical(y_test) , preds))


# In[ ]:


model.save("InceptionV3_Ver1.h5")


# In[ ]:


# It can be used to reconstruct the model identically.
reconstructed_model = tf.keras.models.load_model("InceptionV3_Ver1.h5")

# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:
reconstructed_model.evaluate(test_generator)


# In[ ]:


model.evaluate(test_generator)


# In[ ]:




