import sys
import os
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
 
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.densenet import preprocess_input
 
target_size = (224, 224) 
 
# 预测函数
# 输入：model，图片，目标尺寸
# 输出：预测predict
def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)
 
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]
 
# 画图函数
 
labels = os.listdir('./datas/train')
def plot_preds(image, preds,labels):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
  plt.imshow(image)
  plt.axis('off')
  plt.figure()
  plt.barh([0, 1,2,3], preds, alpha=0.5)
  plt.yticks([0, 1,2,3], labels,fontproperties=font) 
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()
 
# 载入模型
model = load_model('./model/checkpoint-07e-val_acc_0.93.hdf5')
 
# 本地图片进行预测
img = Image.open('./predicts/1.jpg')
preds = predict(model, img, target_size)
plot_preds(img, preds,labels)
