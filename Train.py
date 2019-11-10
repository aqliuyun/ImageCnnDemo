import os
import sys
import glob
import matplotlib.pyplot as plt
import keras
from keras import __version__
from keras.applications.densenet import DenseNet201,preprocess_input
 
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import Callback

class PrintLoss(Callback):
    def on_batch_end(self, batch, logs={}):
        print(logs.get('loss'))

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt
 
 
# 数据准备
IM_WIDTH, IM_HEIGHT = 224, 224 #densenet指定的图片尺寸
 
 
train_dir = './datas/train'  # 训练集数据路径
val_dir = './datas/train' # 验证集数据nb_classes = 4
nb_epoch = 20
nb_batch_size = 32
 
nb_train_samples = get_nb_files(train_dir)      # 训练样本个数
nb_classes = len(glob.glob(train_dir + "/*"))  # 分类数
nb_val_samples = get_nb_files(val_dir)       #验证集样本个数
nb_epoch = int(nb_epoch)                # epoch数量
nb_batch_size = int(nb_batch_size)           
labels = os.listdir('./datas/train')

    
#图片生成器
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
  rotation_range=60,
  width_shift_range=0.2,
  height_shift_range=0.2,
  zoom_range=0.5,
  horizontal_flip=True)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
  rotation_range=60,
  width_shift_range=0.2,
  height_shift_range=0.2,
  zoom_range=0.5,
  horizontal_flip=True)
 
# 训练数据与测试数据
train_generator = train_datagen.flow_from_directory(train_dir,
classes=labels,
#save_to_dir='./result/train/',
#save_prefix='car',
#save_format='jpeg',
target_size=(IM_WIDTH, IM_HEIGHT),
batch_size=nb_batch_size,
interpolation='bilinear',
class_mode='categorical')
 
validation_generator = test_datagen.flow_from_directory(val_dir,
classes=labels,
#save_to_dir='./result/validation/',
#save_prefix='car',
#save_format='jpeg',
target_size=(IM_WIDTH, IM_HEIGHT),
batch_size=nb_batch_size,
interpolation='bilinear',
class_mode='categorical')
 
# 添加新层
def add_new_last_layer(base_model, nb_classes):
  """
  添加最后的层
  输入
  base_model和分类数量
  输出
  新的keras的model
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  #x = Dense(512, activation='relu')(x) #new dense layer
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model

 
#搭建模型
base_model = keras.applications.vgg19.VGG19(include_top=False,weights="imagenet")
model = add_new_last_layer(base_model,nb_classes)
for layer in base_model.layers:
    layer.trainable = False
model.summary()
#model.load_weights('../model/checkpoint-02e-val_acc_0.82.hdf5')
#第二次训练可以接着第一次训练得到的模型接着训练
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(len(model.layers))
 
#更好地保存模型 Save the model after every epoch.
output_model_file = './model/checkpoint-{epoch:02d}e-val_acc_{val_accuracy:.2f}.hdf5'
#keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
#save_best_only=False, save_weights_only=False, mode='auto', period=1)
checkpoint = ModelCheckpoint(output_model_file, monitor='val_accuracy', verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir='.\logs')
reduceLr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
       factor=0.1,
      patience=10,)
 
#开始训练
history_ft = model.fit_generator(train_generator,
epochs=nb_epoch,
steps_per_epoch=10,
validation_data=validation_generator,
callbacks=[tensorboard,checkpoint])
