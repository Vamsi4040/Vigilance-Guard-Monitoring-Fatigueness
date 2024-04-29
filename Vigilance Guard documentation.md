```python
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
```


```python
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50
```


```python

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\ivams\Desktop\AI Projectt\archive",
    shuffle=True,
    image_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
```

    Found 4000 files belonging to 2 classes.
    


```python
#dataset classes
class_names = dataset.class_names
class_names
```




    ['Closed_Eyes', 'Open_Eyes']




```python
#details of the dataset imported
for image_batch , label_batch in dataset.take(1):
#     print(image_batch[0].numpy())
    print(image_batch.shape)
    print(label_batch.numpy())
```

    (32, 256, 256, 3)
    [1 0 1 0 0 1 1 0 0 1 1 1 1 1 0 1 1 1 1 1 0 1 1 0 1 0 0 1 0 0 0 0]
    


```python
#converting floating images to uint8
for image_batch , label_batch in dataset.take(1):
    plt.figure(figsize=(10,10))
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype('uint8'))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")
```


    
![png](output_5_0.png)
    



```python
len(dataset)
```




    125




```python
train_size = 0.8
splitter = len(dataset)*train_size
```


```python
train_ds = dataset.take(54)
len(train_ds)
```




    54




```python
testing_set = dataset.skip(54)
div_t = len(dataset)*0.1
div_t
```




    12.5




```python
vol_set = testing_set.take(6)
```


```python
testing_set = testing_set.skip(6)
```


```python
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
vol_set = vol_set.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
testing_set = testing_set.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
```


```python
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])
```


```python
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])
```


```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

base_model = MobileNetV2(weights='imagenet', include_top=False)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x) 
x = Dense(128, activation='relu')(x) 
predictions = Dense(1, activation='sigmoid')(x) 

model_mobile = models.Model(inputs=base_model.input, outputs=predictions)

model_mobile.compile(optimizer='adam',
              loss='binary_crossentropy', 
              metrics=['accuracy'])


model_mobile.summary()

```

    WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.
    

    Model: "model_2"
    __________________________________________________________________________________________________
     Layer (type)                Output Shape                 Param #   Connected to                  
    ==================================================================================================
     input_3 (InputLayer)        [(None, None, None, 3)]      0         []                            
                                                                                                      
     Conv1 (Conv2D)              (None, None, None, 32)       864       ['input_3[0][0]']             
                                                                                                      
     bn_Conv1 (BatchNormalizati  (None, None, None, 32)       128       ['Conv1[0][0]']               
     on)                                                                                              
                                                                                                      
     Conv1_relu (ReLU)           (None, None, None, 32)       0         ['bn_Conv1[0][0]']            
                                                                                                      
     expanded_conv_depthwise (D  (None, None, None, 32)       288       ['Conv1_relu[0][0]']          
     epthwiseConv2D)                                                                                  
                                                                                                      
     expanded_conv_depthwise_BN  (None, None, None, 32)       128       ['expanded_conv_depthwise[0][0
      (BatchNormalization)                                              ]']                           
                                                                                                      
     expanded_conv_depthwise_re  (None, None, None, 32)       0         ['expanded_conv_depthwise_BN[0
     lu (ReLU)                                                          ][0]']                        
                                                                                                      
     expanded_conv_project (Con  (None, None, None, 16)       512       ['expanded_conv_depthwise_relu
     v2D)                                                               [0][0]']                      
                                                                                                      
     expanded_conv_project_BN (  (None, None, None, 16)       64        ['expanded_conv_project[0][0]'
     BatchNormalization)                                                ]                             
                                                                                                      
     block_1_expand (Conv2D)     (None, None, None, 96)       1536      ['expanded_conv_project_BN[0][
                                                                        0]']                          
                                                                                                      
     block_1_expand_BN (BatchNo  (None, None, None, 96)       384       ['block_1_expand[0][0]']      
     rmalization)                                                                                     
                                                                                                      
     block_1_expand_relu (ReLU)  (None, None, None, 96)       0         ['block_1_expand_BN[0][0]']   
                                                                                                      
     block_1_pad (ZeroPadding2D  (None, None, None, 96)       0         ['block_1_expand_relu[0][0]'] 
     )                                                                                                
                                                                                                      
     block_1_depthwise (Depthwi  (None, None, None, 96)       864       ['block_1_pad[0][0]']         
     seConv2D)                                                                                        
                                                                                                      
     block_1_depthwise_BN (Batc  (None, None, None, 96)       384       ['block_1_depthwise[0][0]']   
     hNormalization)                                                                                  
                                                                                                      
     block_1_depthwise_relu (Re  (None, None, None, 96)       0         ['block_1_depthwise_BN[0][0]']
     LU)                                                                                              
                                                                                                      
     block_1_project (Conv2D)    (None, None, None, 24)       2304      ['block_1_depthwise_relu[0][0]
                                                                        ']                            
                                                                                                      
     block_1_project_BN (BatchN  (None, None, None, 24)       96        ['block_1_project[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_2_expand (Conv2D)     (None, None, None, 144)      3456      ['block_1_project_BN[0][0]']  
                                                                                                      
     block_2_expand_BN (BatchNo  (None, None, None, 144)      576       ['block_2_expand[0][0]']      
     rmalization)                                                                                     
                                                                                                      
     block_2_expand_relu (ReLU)  (None, None, None, 144)      0         ['block_2_expand_BN[0][0]']   
                                                                                                      
     block_2_depthwise (Depthwi  (None, None, None, 144)      1296      ['block_2_expand_relu[0][0]'] 
     seConv2D)                                                                                        
                                                                                                      
     block_2_depthwise_BN (Batc  (None, None, None, 144)      576       ['block_2_depthwise[0][0]']   
     hNormalization)                                                                                  
                                                                                                      
     block_2_depthwise_relu (Re  (None, None, None, 144)      0         ['block_2_depthwise_BN[0][0]']
     LU)                                                                                              
                                                                                                      
     block_2_project (Conv2D)    (None, None, None, 24)       3456      ['block_2_depthwise_relu[0][0]
                                                                        ']                            
                                                                                                      
     block_2_project_BN (BatchN  (None, None, None, 24)       96        ['block_2_project[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_2_add (Add)           (None, None, None, 24)       0         ['block_1_project_BN[0][0]',  
                                                                         'block_2_project_BN[0][0]']  
                                                                                                      
     block_3_expand (Conv2D)     (None, None, None, 144)      3456      ['block_2_add[0][0]']         
                                                                                                      
     block_3_expand_BN (BatchNo  (None, None, None, 144)      576       ['block_3_expand[0][0]']      
     rmalization)                                                                                     
                                                                                                      
     block_3_expand_relu (ReLU)  (None, None, None, 144)      0         ['block_3_expand_BN[0][0]']   
                                                                                                      
     block_3_pad (ZeroPadding2D  (None, None, None, 144)      0         ['block_3_expand_relu[0][0]'] 
     )                                                                                                
                                                                                                      
     block_3_depthwise (Depthwi  (None, None, None, 144)      1296      ['block_3_pad[0][0]']         
     seConv2D)                                                                                        
                                                                                                      
     block_3_depthwise_BN (Batc  (None, None, None, 144)      576       ['block_3_depthwise[0][0]']   
     hNormalization)                                                                                  
                                                                                                      
     block_3_depthwise_relu (Re  (None, None, None, 144)      0         ['block_3_depthwise_BN[0][0]']
     LU)                                                                                              
                                                                                                      
     block_3_project (Conv2D)    (None, None, None, 32)       4608      ['block_3_depthwise_relu[0][0]
                                                                        ']                            
                                                                                                      
     block_3_project_BN (BatchN  (None, None, None, 32)       128       ['block_3_project[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_4_expand (Conv2D)     (None, None, None, 192)      6144      ['block_3_project_BN[0][0]']  
                                                                                                      
     block_4_expand_BN (BatchNo  (None, None, None, 192)      768       ['block_4_expand[0][0]']      
     rmalization)                                                                                     
                                                                                                      
     block_4_expand_relu (ReLU)  (None, None, None, 192)      0         ['block_4_expand_BN[0][0]']   
                                                                                                      
     block_4_depthwise (Depthwi  (None, None, None, 192)      1728      ['block_4_expand_relu[0][0]'] 
     seConv2D)                                                                                        
                                                                                                      
     block_4_depthwise_BN (Batc  (None, None, None, 192)      768       ['block_4_depthwise[0][0]']   
     hNormalization)                                                                                  
                                                                                                      
     block_4_depthwise_relu (Re  (None, None, None, 192)      0         ['block_4_depthwise_BN[0][0]']
     LU)                                                                                              
                                                                                                      
     block_4_project (Conv2D)    (None, None, None, 32)       6144      ['block_4_depthwise_relu[0][0]
                                                                        ']                            
                                                                                                      
     block_4_project_BN (BatchN  (None, None, None, 32)       128       ['block_4_project[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_4_add (Add)           (None, None, None, 32)       0         ['block_3_project_BN[0][0]',  
                                                                         'block_4_project_BN[0][0]']  
                                                                                                      
     block_5_expand (Conv2D)     (None, None, None, 192)      6144      ['block_4_add[0][0]']         
                                                                                                      
     block_5_expand_BN (BatchNo  (None, None, None, 192)      768       ['block_5_expand[0][0]']      
     rmalization)                                                                                     
                                                                                                      
     block_5_expand_relu (ReLU)  (None, None, None, 192)      0         ['block_5_expand_BN[0][0]']   
                                                                                                      
     block_5_depthwise (Depthwi  (None, None, None, 192)      1728      ['block_5_expand_relu[0][0]'] 
     seConv2D)                                                                                        
                                                                                                      
     block_5_depthwise_BN (Batc  (None, None, None, 192)      768       ['block_5_depthwise[0][0]']   
     hNormalization)                                                                                  
                                                                                                      
     block_5_depthwise_relu (Re  (None, None, None, 192)      0         ['block_5_depthwise_BN[0][0]']
     LU)                                                                                              
                                                                                                      
     block_5_project (Conv2D)    (None, None, None, 32)       6144      ['block_5_depthwise_relu[0][0]
                                                                        ']                            
                                                                                                      
     block_5_project_BN (BatchN  (None, None, None, 32)       128       ['block_5_project[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_5_add (Add)           (None, None, None, 32)       0         ['block_4_add[0][0]',         
                                                                         'block_5_project_BN[0][0]']  
                                                                                                      
     block_6_expand (Conv2D)     (None, None, None, 192)      6144      ['block_5_add[0][0]']         
                                                                                                      
     block_6_expand_BN (BatchNo  (None, None, None, 192)      768       ['block_6_expand[0][0]']      
     rmalization)                                                                                     
                                                                                                      
     block_6_expand_relu (ReLU)  (None, None, None, 192)      0         ['block_6_expand_BN[0][0]']   
                                                                                                      
     block_6_pad (ZeroPadding2D  (None, None, None, 192)      0         ['block_6_expand_relu[0][0]'] 
     )                                                                                                
                                                                                                      
     block_6_depthwise (Depthwi  (None, None, None, 192)      1728      ['block_6_pad[0][0]']         
     seConv2D)                                                                                        
                                                                                                      
     block_6_depthwise_BN (Batc  (None, None, None, 192)      768       ['block_6_depthwise[0][0]']   
     hNormalization)                                                                                  
                                                                                                      
     block_6_depthwise_relu (Re  (None, None, None, 192)      0         ['block_6_depthwise_BN[0][0]']
     LU)                                                                                              
                                                                                                      
     block_6_project (Conv2D)    (None, None, None, 64)       12288     ['block_6_depthwise_relu[0][0]
                                                                        ']                            
                                                                                                      
     block_6_project_BN (BatchN  (None, None, None, 64)       256       ['block_6_project[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_7_expand (Conv2D)     (None, None, None, 384)      24576     ['block_6_project_BN[0][0]']  
                                                                                                      
     block_7_expand_BN (BatchNo  (None, None, None, 384)      1536      ['block_7_expand[0][0]']      
     rmalization)                                                                                     
                                                                                                      
     block_7_expand_relu (ReLU)  (None, None, None, 384)      0         ['block_7_expand_BN[0][0]']   
                                                                                                      
     block_7_depthwise (Depthwi  (None, None, None, 384)      3456      ['block_7_expand_relu[0][0]'] 
     seConv2D)                                                                                        
                                                                                                      
     block_7_depthwise_BN (Batc  (None, None, None, 384)      1536      ['block_7_depthwise[0][0]']   
     hNormalization)                                                                                  
                                                                                                      
     block_7_depthwise_relu (Re  (None, None, None, 384)      0         ['block_7_depthwise_BN[0][0]']
     LU)                                                                                              
                                                                                                      
     block_7_project (Conv2D)    (None, None, None, 64)       24576     ['block_7_depthwise_relu[0][0]
                                                                        ']                            
                                                                                                      
     block_7_project_BN (BatchN  (None, None, None, 64)       256       ['block_7_project[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_7_add (Add)           (None, None, None, 64)       0         ['block_6_project_BN[0][0]',  
                                                                         'block_7_project_BN[0][0]']  
                                                                                                      
     block_8_expand (Conv2D)     (None, None, None, 384)      24576     ['block_7_add[0][0]']         
                                                                                                      
     block_8_expand_BN (BatchNo  (None, None, None, 384)      1536      ['block_8_expand[0][0]']      
     rmalization)                                                                                     
                                                                                                      
     block_8_expand_relu (ReLU)  (None, None, None, 384)      0         ['block_8_expand_BN[0][0]']   
                                                                                                      
     block_8_depthwise (Depthwi  (None, None, None, 384)      3456      ['block_8_expand_relu[0][0]'] 
     seConv2D)                                                                                        
                                                                                                      
     block_8_depthwise_BN (Batc  (None, None, None, 384)      1536      ['block_8_depthwise[0][0]']   
     hNormalization)                                                                                  
                                                                                                      
     block_8_depthwise_relu (Re  (None, None, None, 384)      0         ['block_8_depthwise_BN[0][0]']
     LU)                                                                                              
                                                                                                      
     block_8_project (Conv2D)    (None, None, None, 64)       24576     ['block_8_depthwise_relu[0][0]
                                                                        ']                            
                                                                                                      
     block_8_project_BN (BatchN  (None, None, None, 64)       256       ['block_8_project[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_8_add (Add)           (None, None, None, 64)       0         ['block_7_add[0][0]',         
                                                                         'block_8_project_BN[0][0]']  
                                                                                                      
     block_9_expand (Conv2D)     (None, None, None, 384)      24576     ['block_8_add[0][0]']         
                                                                                                      
     block_9_expand_BN (BatchNo  (None, None, None, 384)      1536      ['block_9_expand[0][0]']      
     rmalization)                                                                                     
                                                                                                      
     block_9_expand_relu (ReLU)  (None, None, None, 384)      0         ['block_9_expand_BN[0][0]']   
                                                                                                      
     block_9_depthwise (Depthwi  (None, None, None, 384)      3456      ['block_9_expand_relu[0][0]'] 
     seConv2D)                                                                                        
                                                                                                      
     block_9_depthwise_BN (Batc  (None, None, None, 384)      1536      ['block_9_depthwise[0][0]']   
     hNormalization)                                                                                  
                                                                                                      
     block_9_depthwise_relu (Re  (None, None, None, 384)      0         ['block_9_depthwise_BN[0][0]']
     LU)                                                                                              
                                                                                                      
     block_9_project (Conv2D)    (None, None, None, 64)       24576     ['block_9_depthwise_relu[0][0]
                                                                        ']                            
                                                                                                      
     block_9_project_BN (BatchN  (None, None, None, 64)       256       ['block_9_project[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_9_add (Add)           (None, None, None, 64)       0         ['block_8_add[0][0]',         
                                                                         'block_9_project_BN[0][0]']  
                                                                                                      
     block_10_expand (Conv2D)    (None, None, None, 384)      24576     ['block_9_add[0][0]']         
                                                                                                      
     block_10_expand_BN (BatchN  (None, None, None, 384)      1536      ['block_10_expand[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_10_expand_relu (ReLU  (None, None, None, 384)      0         ['block_10_expand_BN[0][0]']  
     )                                                                                                
                                                                                                      
     block_10_depthwise (Depthw  (None, None, None, 384)      3456      ['block_10_expand_relu[0][0]']
     iseConv2D)                                                                                       
                                                                                                      
     block_10_depthwise_BN (Bat  (None, None, None, 384)      1536      ['block_10_depthwise[0][0]']  
     chNormalization)                                                                                 
                                                                                                      
     block_10_depthwise_relu (R  (None, None, None, 384)      0         ['block_10_depthwise_BN[0][0]'
     eLU)                                                               ]                             
                                                                                                      
     block_10_project (Conv2D)   (None, None, None, 96)       36864     ['block_10_depthwise_relu[0][0
                                                                        ]']                           
                                                                                                      
     block_10_project_BN (Batch  (None, None, None, 96)       384       ['block_10_project[0][0]']    
     Normalization)                                                                                   
                                                                                                      
     block_11_expand (Conv2D)    (None, None, None, 576)      55296     ['block_10_project_BN[0][0]'] 
                                                                                                      
     block_11_expand_BN (BatchN  (None, None, None, 576)      2304      ['block_11_expand[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_11_expand_relu (ReLU  (None, None, None, 576)      0         ['block_11_expand_BN[0][0]']  
     )                                                                                                
                                                                                                      
     block_11_depthwise (Depthw  (None, None, None, 576)      5184      ['block_11_expand_relu[0][0]']
     iseConv2D)                                                                                       
                                                                                                      
     block_11_depthwise_BN (Bat  (None, None, None, 576)      2304      ['block_11_depthwise[0][0]']  
     chNormalization)                                                                                 
                                                                                                      
     block_11_depthwise_relu (R  (None, None, None, 576)      0         ['block_11_depthwise_BN[0][0]'
     eLU)                                                               ]                             
                                                                                                      
     block_11_project (Conv2D)   (None, None, None, 96)       55296     ['block_11_depthwise_relu[0][0
                                                                        ]']                           
                                                                                                      
     block_11_project_BN (Batch  (None, None, None, 96)       384       ['block_11_project[0][0]']    
     Normalization)                                                                                   
                                                                                                      
     block_11_add (Add)          (None, None, None, 96)       0         ['block_10_project_BN[0][0]', 
                                                                         'block_11_project_BN[0][0]'] 
                                                                                                      
     block_12_expand (Conv2D)    (None, None, None, 576)      55296     ['block_11_add[0][0]']        
                                                                                                      
     block_12_expand_BN (BatchN  (None, None, None, 576)      2304      ['block_12_expand[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_12_expand_relu (ReLU  (None, None, None, 576)      0         ['block_12_expand_BN[0][0]']  
     )                                                                                                
                                                                                                      
     block_12_depthwise (Depthw  (None, None, None, 576)      5184      ['block_12_expand_relu[0][0]']
     iseConv2D)                                                                                       
                                                                                                      
     block_12_depthwise_BN (Bat  (None, None, None, 576)      2304      ['block_12_depthwise[0][0]']  
     chNormalization)                                                                                 
                                                                                                      
     block_12_depthwise_relu (R  (None, None, None, 576)      0         ['block_12_depthwise_BN[0][0]'
     eLU)                                                               ]                             
                                                                                                      
     block_12_project (Conv2D)   (None, None, None, 96)       55296     ['block_12_depthwise_relu[0][0
                                                                        ]']                           
                                                                                                      
     block_12_project_BN (Batch  (None, None, None, 96)       384       ['block_12_project[0][0]']    
     Normalization)                                                                                   
                                                                                                      
     block_12_add (Add)          (None, None, None, 96)       0         ['block_11_add[0][0]',        
                                                                         'block_12_project_BN[0][0]'] 
                                                                                                      
     block_13_expand (Conv2D)    (None, None, None, 576)      55296     ['block_12_add[0][0]']        
                                                                                                      
     block_13_expand_BN (BatchN  (None, None, None, 576)      2304      ['block_13_expand[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_13_expand_relu (ReLU  (None, None, None, 576)      0         ['block_13_expand_BN[0][0]']  
     )                                                                                                
                                                                                                      
     block_13_pad (ZeroPadding2  (None, None, None, 576)      0         ['block_13_expand_relu[0][0]']
     D)                                                                                               
                                                                                                      
     block_13_depthwise (Depthw  (None, None, None, 576)      5184      ['block_13_pad[0][0]']        
     iseConv2D)                                                                                       
                                                                                                      
     block_13_depthwise_BN (Bat  (None, None, None, 576)      2304      ['block_13_depthwise[0][0]']  
     chNormalization)                                                                                 
                                                                                                      
     block_13_depthwise_relu (R  (None, None, None, 576)      0         ['block_13_depthwise_BN[0][0]'
     eLU)                                                               ]                             
                                                                                                      
     block_13_project (Conv2D)   (None, None, None, 160)      92160     ['block_13_depthwise_relu[0][0
                                                                        ]']                           
                                                                                                      
     block_13_project_BN (Batch  (None, None, None, 160)      640       ['block_13_project[0][0]']    
     Normalization)                                                                                   
                                                                                                      
     block_14_expand (Conv2D)    (None, None, None, 960)      153600    ['block_13_project_BN[0][0]'] 
                                                                                                      
     block_14_expand_BN (BatchN  (None, None, None, 960)      3840      ['block_14_expand[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_14_expand_relu (ReLU  (None, None, None, 960)      0         ['block_14_expand_BN[0][0]']  
     )                                                                                                
                                                                                                      
     block_14_depthwise (Depthw  (None, None, None, 960)      8640      ['block_14_expand_relu[0][0]']
     iseConv2D)                                                                                       
                                                                                                      
     block_14_depthwise_BN (Bat  (None, None, None, 960)      3840      ['block_14_depthwise[0][0]']  
     chNormalization)                                                                                 
                                                                                                      
     block_14_depthwise_relu (R  (None, None, None, 960)      0         ['block_14_depthwise_BN[0][0]'
     eLU)                                                               ]                             
                                                                                                      
     block_14_project (Conv2D)   (None, None, None, 160)      153600    ['block_14_depthwise_relu[0][0
                                                                        ]']                           
                                                                                                      
     block_14_project_BN (Batch  (None, None, None, 160)      640       ['block_14_project[0][0]']    
     Normalization)                                                                                   
                                                                                                      
     block_14_add (Add)          (None, None, None, 160)      0         ['block_13_project_BN[0][0]', 
                                                                         'block_14_project_BN[0][0]'] 
                                                                                                      
     block_15_expand (Conv2D)    (None, None, None, 960)      153600    ['block_14_add[0][0]']        
                                                                                                      
     block_15_expand_BN (BatchN  (None, None, None, 960)      3840      ['block_15_expand[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_15_expand_relu (ReLU  (None, None, None, 960)      0         ['block_15_expand_BN[0][0]']  
     )                                                                                                
                                                                                                      
     block_15_depthwise (Depthw  (None, None, None, 960)      8640      ['block_15_expand_relu[0][0]']
     iseConv2D)                                                                                       
                                                                                                      
     block_15_depthwise_BN (Bat  (None, None, None, 960)      3840      ['block_15_depthwise[0][0]']  
     chNormalization)                                                                                 
                                                                                                      
     block_15_depthwise_relu (R  (None, None, None, 960)      0         ['block_15_depthwise_BN[0][0]'
     eLU)                                                               ]                             
                                                                                                      
     block_15_project (Conv2D)   (None, None, None, 160)      153600    ['block_15_depthwise_relu[0][0
                                                                        ]']                           
                                                                                                      
     block_15_project_BN (Batch  (None, None, None, 160)      640       ['block_15_project[0][0]']    
     Normalization)                                                                                   
                                                                                                      
     block_15_add (Add)          (None, None, None, 160)      0         ['block_14_add[0][0]',        
                                                                         'block_15_project_BN[0][0]'] 
                                                                                                      
     block_16_expand (Conv2D)    (None, None, None, 960)      153600    ['block_15_add[0][0]']        
                                                                                                      
     block_16_expand_BN (BatchN  (None, None, None, 960)      3840      ['block_16_expand[0][0]']     
     ormalization)                                                                                    
                                                                                                      
     block_16_expand_relu (ReLU  (None, None, None, 960)      0         ['block_16_expand_BN[0][0]']  
     )                                                                                                
                                                                                                      
     block_16_depthwise (Depthw  (None, None, None, 960)      8640      ['block_16_expand_relu[0][0]']
     iseConv2D)                                                                                       
                                                                                                      
     block_16_depthwise_BN (Bat  (None, None, None, 960)      3840      ['block_16_depthwise[0][0]']  
     chNormalization)                                                                                 
                                                                                                      
     block_16_depthwise_relu (R  (None, None, None, 960)      0         ['block_16_depthwise_BN[0][0]'
     eLU)                                                               ]                             
                                                                                                      
     block_16_project (Conv2D)   (None, None, None, 320)      307200    ['block_16_depthwise_relu[0][0
                                                                        ]']                           
                                                                                                      
     block_16_project_BN (Batch  (None, None, None, 320)      1280      ['block_16_project[0][0]']    
     Normalization)                                                                                   
                                                                                                      
     Conv_1 (Conv2D)             (None, None, None, 1280)     409600    ['block_16_project_BN[0][0]'] 
                                                                                                      
     Conv_1_bn (BatchNormalizat  (None, None, None, 1280)     5120      ['Conv_1[0][0]']              
     ion)                                                                                             
                                                                                                      
     out_relu (ReLU)             (None, None, None, 1280)     0         ['Conv_1_bn[0][0]']           
                                                                                                      
     global_average_pooling2d_2  (None, 1280)                 0         ['out_relu[0][0]']            
      (GlobalAveragePooling2D)                                                                        
                                                                                                      
     dense_8 (Dense)             (None, 128)                  163968    ['global_average_pooling2d_2[0
                                                                        ][0]']                        
                                                                                                      
     dense_9 (Dense)             (None, 1)                    129       ['dense_8[0][0]']             
                                                                                                      
    ==================================================================================================
    Total params: 2422081 (9.24 MB)
    Trainable params: 164097 (641.00 KB)
    Non-trainable params: 2257984 (8.61 MB)
    __________________________________________________________________________________________________
    


```python
tf.keras.utils.plot_model(model_mobile, to_file='model_architecture.png', show_shapes=True)
```




    
![png](output_16_0.png)
    




```python
history = model_mobile.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size = BATCH_SIZE,
    verbose = 1,
    validation_data=vol_set
)
```

    Epoch 1/50
    54/54 [==============================] - 17s 171ms/step - loss: 0.4162 - accuracy: 0.7917 - val_loss: 0.2743 - val_accuracy: 0.9010
    Epoch 2/50
    54/54 [==============================] - 4s 68ms/step - loss: 0.2509 - accuracy: 0.9074 - val_loss: 0.2086 - val_accuracy: 0.9167
    Epoch 3/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.1787 - accuracy: 0.9416 - val_loss: 0.1578 - val_accuracy: 0.9583
    Epoch 4/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.1377 - accuracy: 0.9578 - val_loss: 0.1102 - val_accuracy: 0.9583
    Epoch 5/50
    54/54 [==============================] - 4s 71ms/step - loss: 0.1142 - accuracy: 0.9624 - val_loss: 0.0988 - val_accuracy: 0.9635
    Epoch 6/50
    54/54 [==============================] - 4s 74ms/step - loss: 0.0902 - accuracy: 0.9676 - val_loss: 0.0723 - val_accuracy: 0.9740
    Epoch 7/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0921 - accuracy: 0.9659 - val_loss: 0.0812 - val_accuracy: 0.9635
    Epoch 8/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0670 - accuracy: 0.9809 - val_loss: 0.0593 - val_accuracy: 0.9740
    Epoch 9/50
    54/54 [==============================] - 4s 73ms/step - loss: 0.0679 - accuracy: 0.9792 - val_loss: 0.0483 - val_accuracy: 0.9844
    Epoch 10/50
    54/54 [==============================] - 4s 67ms/step - loss: 0.0537 - accuracy: 0.9838 - val_loss: 0.0431 - val_accuracy: 0.9844
    Epoch 11/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0473 - accuracy: 0.9844 - val_loss: 0.0339 - val_accuracy: 1.0000
    Epoch 12/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0393 - accuracy: 0.9925 - val_loss: 0.0288 - val_accuracy: 0.9948
    Epoch 13/50
    54/54 [==============================] - 4s 70ms/step - loss: 0.0330 - accuracy: 0.9925 - val_loss: 0.0254 - val_accuracy: 1.0000
    Epoch 14/50
    54/54 [==============================] - 4s 68ms/step - loss: 0.0312 - accuracy: 0.9942 - val_loss: 0.0231 - val_accuracy: 1.0000
    Epoch 15/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0449 - accuracy: 0.9844 - val_loss: 0.0568 - val_accuracy: 0.9688
    Epoch 16/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0278 - accuracy: 0.9942 - val_loss: 0.0296 - val_accuracy: 0.9844
    Epoch 17/50
    54/54 [==============================] - 3s 65ms/step - loss: 0.0217 - accuracy: 0.9954 - val_loss: 0.0198 - val_accuracy: 0.9948
    Epoch 18/50
    54/54 [==============================] - 4s 74ms/step - loss: 0.0248 - accuracy: 0.9936 - val_loss: 0.0203 - val_accuracy: 0.9896
    Epoch 19/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0268 - accuracy: 0.9936 - val_loss: 0.0149 - val_accuracy: 1.0000
    Epoch 20/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0217 - accuracy: 0.9942 - val_loss: 0.0179 - val_accuracy: 0.9948
    Epoch 21/50
    54/54 [==============================] - 4s 73ms/step - loss: 0.0143 - accuracy: 0.9988 - val_loss: 0.0115 - val_accuracy: 1.0000
    Epoch 22/50
    54/54 [==============================] - 4s 67ms/step - loss: 0.0138 - accuracy: 0.9988 - val_loss: 0.0133 - val_accuracy: 1.0000
    Epoch 23/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0167 - accuracy: 0.9965 - val_loss: 0.0140 - val_accuracy: 1.0000
    Epoch 24/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0142 - accuracy: 0.9988 - val_loss: 0.0141 - val_accuracy: 1.0000
    Epoch 25/50
    54/54 [==============================] - 4s 72ms/step - loss: 0.0103 - accuracy: 1.0000 - val_loss: 0.0090 - val_accuracy: 1.0000
    Epoch 26/50
    54/54 [==============================] - 4s 69ms/step - loss: 0.0094 - accuracy: 0.9994 - val_loss: 0.0081 - val_accuracy: 1.0000
    Epoch 27/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0088 - accuracy: 0.9988 - val_loss: 0.0115 - val_accuracy: 1.0000
    Epoch 28/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0079 - accuracy: 0.9994 - val_loss: 0.0139 - val_accuracy: 0.9948
    Epoch 29/50
    54/54 [==============================] - 4s 71ms/step - loss: 0.0079 - accuracy: 1.0000 - val_loss: 0.0070 - val_accuracy: 1.0000
    Epoch 30/50
    54/54 [==============================] - 4s 66ms/step - loss: 0.0084 - accuracy: 0.9994 - val_loss: 0.0073 - val_accuracy: 1.0000
    Epoch 31/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0072 - accuracy: 1.0000 - val_loss: 0.0081 - val_accuracy: 1.0000
    Epoch 32/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0060 - accuracy: 1.0000 - val_loss: 0.0070 - val_accuracy: 1.0000
    Epoch 33/50
    54/54 [==============================] - 4s 68ms/step - loss: 0.0054 - accuracy: 1.0000 - val_loss: 0.0079 - val_accuracy: 1.0000
    Epoch 34/50
    54/54 [==============================] - 4s 69ms/step - loss: 0.0054 - accuracy: 1.0000 - val_loss: 0.0058 - val_accuracy: 1.0000
    Epoch 35/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0052 - accuracy: 1.0000 - val_loss: 0.0136 - val_accuracy: 0.9948
    Epoch 36/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0053 - accuracy: 1.0000 - val_loss: 0.0044 - val_accuracy: 1.0000
    Epoch 37/50
    54/54 [==============================] - 4s 72ms/step - loss: 0.0103 - accuracy: 0.9977 - val_loss: 0.0074 - val_accuracy: 1.0000
    Epoch 38/50
    54/54 [==============================] - 4s 68ms/step - loss: 0.0138 - accuracy: 0.9959 - val_loss: 0.0272 - val_accuracy: 0.9896
    Epoch 39/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0091 - accuracy: 0.9983 - val_loss: 0.0040 - val_accuracy: 1.0000
    Epoch 40/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0031 - accuracy: 1.0000 - val_loss: 0.0035 - val_accuracy: 1.0000
    Epoch 41/50
    54/54 [==============================] - 3s 65ms/step - loss: 0.0041 - accuracy: 1.0000 - val_loss: 0.0033 - val_accuracy: 1.0000
    Epoch 42/50
    54/54 [==============================] - 4s 74ms/step - loss: 0.0089 - accuracy: 0.9983 - val_loss: 0.0050 - val_accuracy: 1.0000
    Epoch 43/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0035 - accuracy: 1.0000 - val_loss: 0.0030 - val_accuracy: 1.0000
    Epoch 44/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0026 - accuracy: 1.0000 - val_loss: 0.0038 - val_accuracy: 1.0000
    Epoch 45/50
    54/54 [==============================] - 4s 71ms/step - loss: 0.0024 - accuracy: 1.0000 - val_loss: 0.0026 - val_accuracy: 1.0000
    Epoch 46/50
    54/54 [==============================] - 4s 73ms/step - loss: 0.0028 - accuracy: 1.0000 - val_loss: 0.0029 - val_accuracy: 1.0000
    Epoch 47/50
    54/54 [==============================] - 4s 65ms/step - loss: 0.0019 - accuracy: 1.0000 - val_loss: 0.0032 - val_accuracy: 1.0000
    Epoch 48/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0019 - accuracy: 1.0000 - val_loss: 0.0047 - val_accuracy: 1.0000
    Epoch 49/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0019 - accuracy: 1.0000 - val_loss: 0.0026 - val_accuracy: 1.0000
    Epoch 50/50
    54/54 [==============================] - 4s 74ms/step - loss: 0.0027 - accuracy: 1.0000 - val_loss: 0.0063 - val_accuracy: 1.0000
    


```python
model_mobile.save("model_mobile.h5")
```

    /usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3079: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
      saving_api.save_model(
    


```python
input_shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
model = models.Sequential([
    resize_and_rescale,
    data_augmentation,

    layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE)),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPool2D((2,2)),



    layers.Flatten(),
    layers.Dense(64),
    layers.Dense(1, activation="sigmoid"),

])
```


```python
model.build(input_shape=input_shape)
```


```python
model.summary()
```

    Model: "sequential_10"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     sequential_6 (Sequential)   (None, 256, 256, 3)       0         
                                                                     
     sequential_7 (Sequential)   (None, 256, 256, 3)       0         
                                                                     
     conv2d_24 (Conv2D)          (32, 254, 254, 32)        896       
                                                                     
     max_pooling2d_24 (MaxPooli  (32, 127, 127, 32)        0         
     ng2D)                                                           
                                                                     
     conv2d_25 (Conv2D)          (32, 125, 125, 64)        18496     
                                                                     
     max_pooling2d_25 (MaxPooli  (32, 62, 62, 64)          0         
     ng2D)                                                           
                                                                     
     conv2d_26 (Conv2D)          (32, 60, 60, 64)          36928     
                                                                     
     max_pooling2d_26 (MaxPooli  (32, 30, 30, 64)          0         
     ng2D)                                                           
                                                                     
     conv2d_27 (Conv2D)          (32, 28, 28, 64)          36928     
                                                                     
     max_pooling2d_27 (MaxPooli  (32, 14, 14, 64)          0         
     ng2D)                                                           
                                                                     
     conv2d_28 (Conv2D)          (32, 12, 12, 64)          36928     
                                                                     
     max_pooling2d_28 (MaxPooli  (32, 6, 6, 64)            0         
     ng2D)                                                           
                                                                     
     conv2d_29 (Conv2D)          (32, 4, 4, 64)            36928     
                                                                     
     max_pooling2d_29 (MaxPooli  (32, 2, 2, 64)            0         
     ng2D)                                                           
                                                                     
     flatten_4 (Flatten)         (32, 256)                 0         
                                                                     
     dense_14 (Dense)            (32, 64)                  16448     
                                                                     
     dense_15 (Dense)            (32, 1)                   65        
                                                                     
    =================================================================
    Total params: 183617 (717.25 KB)
    Trainable params: 183617 (717.25 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    


```python
tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True)
```




    
![png](output_22_0.png)
    




```python
model.compile(optimizer='Adam',
             loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"]
             )
```


```python
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size = BATCH_SIZE,
    verbose = 1,
    validation_data=vol_set
)
```

    Epoch 1/50
    54/54 [==============================] - 8s 69ms/step - loss: 0.4905 - accuracy: 0.7581 - val_loss: 0.0839 - val_accuracy: 0.9740
    Epoch 2/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.1190 - accuracy: 0.9578 - val_loss: 0.0956 - val_accuracy: 0.9792
    Epoch 3/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.1108 - accuracy: 0.9601 - val_loss: 0.0648 - val_accuracy: 0.9688
    Epoch 4/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0749 - accuracy: 0.9745 - val_loss: 0.1044 - val_accuracy: 0.9479
    Epoch 5/50
    54/54 [==============================] - 4s 66ms/step - loss: 0.0510 - accuracy: 0.9826 - val_loss: 0.0514 - val_accuracy: 0.9896
    Epoch 6/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0420 - accuracy: 0.9861 - val_loss: 0.0359 - val_accuracy: 0.9844
    Epoch 7/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0288 - accuracy: 0.9902 - val_loss: 0.0293 - val_accuracy: 0.9844
    Epoch 8/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0280 - accuracy: 0.9896 - val_loss: 0.0452 - val_accuracy: 0.9896
    Epoch 9/50
    54/54 [==============================] - 4s 66ms/step - loss: 0.0399 - accuracy: 0.9884 - val_loss: 0.0678 - val_accuracy: 0.9844
    Epoch 10/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0327 - accuracy: 0.9878 - val_loss: 0.0305 - val_accuracy: 0.9896
    Epoch 11/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0267 - accuracy: 0.9913 - val_loss: 0.0282 - val_accuracy: 0.9896
    Epoch 12/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0263 - accuracy: 0.9919 - val_loss: 0.0233 - val_accuracy: 0.9896
    Epoch 13/50
    54/54 [==============================] - 4s 66ms/step - loss: 0.0197 - accuracy: 0.9942 - val_loss: 0.0534 - val_accuracy: 0.9844
    Epoch 14/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0140 - accuracy: 0.9948 - val_loss: 0.0298 - val_accuracy: 0.9896
    Epoch 15/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0241 - accuracy: 0.9925 - val_loss: 0.0455 - val_accuracy: 0.9844
    Epoch 16/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0177 - accuracy: 0.9936 - val_loss: 0.0617 - val_accuracy: 0.9844
    Epoch 17/50
    54/54 [==============================] - 3s 65ms/step - loss: 0.0150 - accuracy: 0.9959 - val_loss: 0.0292 - val_accuracy: 0.9896
    Epoch 18/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0135 - accuracy: 0.9948 - val_loss: 0.0410 - val_accuracy: 0.9844
    Epoch 19/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0151 - accuracy: 0.9965 - val_loss: 0.0427 - val_accuracy: 0.9948
    Epoch 20/50
    54/54 [==============================] - 3s 65ms/step - loss: 0.0406 - accuracy: 0.9878 - val_loss: 0.0200 - val_accuracy: 0.9896
    Epoch 21/50
    54/54 [==============================] - 4s 65ms/step - loss: 0.0162 - accuracy: 0.9965 - val_loss: 0.1604 - val_accuracy: 0.9531
    Epoch 22/50
    54/54 [==============================] - 3s 62ms/step - loss: 0.0320 - accuracy: 0.9931 - val_loss: 0.0384 - val_accuracy: 0.9896
    Epoch 23/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0168 - accuracy: 0.9965 - val_loss: 0.0302 - val_accuracy: 0.9948
    Epoch 24/50
    54/54 [==============================] - 4s 66ms/step - loss: 0.0137 - accuracy: 0.9954 - val_loss: 0.0177 - val_accuracy: 0.9948
    Epoch 25/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0148 - accuracy: 0.9948 - val_loss: 0.0022 - val_accuracy: 1.0000
    Epoch 26/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0249 - accuracy: 0.9919 - val_loss: 0.0466 - val_accuracy: 0.9896
    Epoch 27/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0122 - accuracy: 0.9959 - val_loss: 0.0310 - val_accuracy: 0.9948
    Epoch 28/50
    54/54 [==============================] - 3s 65ms/step - loss: 0.0114 - accuracy: 0.9954 - val_loss: 0.0145 - val_accuracy: 0.9948
    Epoch 29/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0141 - accuracy: 0.9942 - val_loss: 0.0029 - val_accuracy: 1.0000
    Epoch 30/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0045 - accuracy: 0.9977 - val_loss: 0.0010 - val_accuracy: 1.0000
    Epoch 31/50
    54/54 [==============================] - 3s 62ms/step - loss: 0.0077 - accuracy: 0.9977 - val_loss: 0.0021 - val_accuracy: 1.0000
    Epoch 32/50
    54/54 [==============================] - 3s 65ms/step - loss: 0.0103 - accuracy: 0.9954 - val_loss: 0.0038 - val_accuracy: 1.0000
    Epoch 33/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0265 - accuracy: 0.9919 - val_loss: 0.0152 - val_accuracy: 0.9948
    Epoch 34/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0194 - accuracy: 0.9942 - val_loss: 0.0506 - val_accuracy: 0.9792
    Epoch 35/50
    54/54 [==============================] - 3s 62ms/step - loss: 0.0074 - accuracy: 0.9977 - val_loss: 0.0160 - val_accuracy: 0.9948
    Epoch 36/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0028 - accuracy: 0.9994 - val_loss: 0.0191 - val_accuracy: 0.9948
    Epoch 37/50
    54/54 [==============================] - 4s 65ms/step - loss: 3.7564e-04 - accuracy: 1.0000 - val_loss: 0.0399 - val_accuracy: 0.9948
    Epoch 38/50
    54/54 [==============================] - 3s 62ms/step - loss: 7.7695e-04 - accuracy: 1.0000 - val_loss: 0.0017 - val_accuracy: 1.0000
    Epoch 39/50
    54/54 [==============================] - 3s 64ms/step - loss: 0.0021 - accuracy: 0.9988 - val_loss: 0.0737 - val_accuracy: 0.9896
    Epoch 40/50
    54/54 [==============================] - 4s 65ms/step - loss: 0.0051 - accuracy: 0.9977 - val_loss: 0.0031 - val_accuracy: 1.0000
    Epoch 41/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0507 - accuracy: 0.9919 - val_loss: 0.0117 - val_accuracy: 0.9948
    Epoch 42/50
    54/54 [==============================] - 3s 62ms/step - loss: 0.0516 - accuracy: 0.9861 - val_loss: 0.0412 - val_accuracy: 0.9896
    Epoch 43/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0078 - accuracy: 0.9965 - val_loss: 0.0334 - val_accuracy: 0.9948
    Epoch 44/50
    54/54 [==============================] - 3s 65ms/step - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.0306 - val_accuracy: 0.9948
    Epoch 45/50
    54/54 [==============================] - 4s 65ms/step - loss: 7.9848e-04 - accuracy: 1.0000 - val_loss: 0.0095 - val_accuracy: 0.9948
    Epoch 46/50
    54/54 [==============================] - 3s 62ms/step - loss: 0.0210 - accuracy: 0.9965 - val_loss: 0.0056 - val_accuracy: 1.0000
    Epoch 47/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0075 - accuracy: 0.9983 - val_loss: 0.0129 - val_accuracy: 0.9896
    Epoch 48/50
    54/54 [==============================] - 3s 63ms/step - loss: 0.0090 - accuracy: 0.9965 - val_loss: 0.0015 - val_accuracy: 1.0000
    Epoch 49/50
    54/54 [==============================] - 4s 65ms/step - loss: 0.0042 - accuracy: 0.9983 - val_loss: 0.0259 - val_accuracy: 0.9948
    Epoch 50/50
    54/54 [==============================] - 3s 62ms/step - loss: 0.0026 - accuracy: 0.9988 - val_loss: 0.0218 - val_accuracy: 0.9948
    


```python
model.save('model.h5')
```


```python
model = tf.keras.models.load_model("model.h5")
```


```python
model.evaluate(testing_set)
```

    65/65 [==============================] - 2s 22ms/step - loss: 0.0365 - accuracy: 0.9909
    




    [0.03646264225244522, 0.9908654093742371]




```python
history.params
acc = history.history['accuracy']
accuracy_per = history.history['val_accuracy']

loss = history.history['loss']
per_l = history.history['val_loss']
```


```python

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), acc, label="train_acc")
plt.plot(range(EPOCHS), accuracy_per, label="label_acc")
plt.legend(loc="lower right")
plt.title("Plotting the training accuracy and validation accuracy")

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label="train_loss")
plt.plot(range(EPOCHS), per_l, label="val_loss")
plt.legend(loc="lower right")
plt.title("plotting the trian and val loss")
```




    Text(0.5, 1.0, 'plotting the trian and val loss')




    
![png](output_29_1.png)
    



```python
model = tf.keras.models.load_model("./model.h5")
```


```python
class_names
```




    ['Closed_Eyes', 'Open_Eyes']




```python

plt.figure(figsize=(10,10))
for image_batch, label_batch in testing_set.take(1):
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))

        actual_label = class_names[label_batch[i]]

        pred_label = model.predict(image_batch)
        print("pred_label", pred_label[i])

        print("pred_label", pred_label[i])

        pred_label_conf = round(np.max(pred_label)*100, 2)
        pred_label = class_names[0 if pred_label[i] < 0.5 else 1]
        print(pred_label)

        if actual_label == pred_label:
            plt.title(f"Actual {actual_label} \n predicted label {pred_label} \nconfidence {pred_label_conf}%" ,color="green")
        else:
            plt.title(f"Actual {actual_label} \n predicted label {pred_label} \nconfidence {pred_label_conf}%", color="red")
```

    1/1 [==============================] - 0s 190ms/step
    pred_label [0.00067054]
    pred_label [0.00067054]
    Closed_Eyes
    1/1 [==============================] - 0s 40ms/step
    pred_label [1.]
    pred_label [1.]
    Open_Eyes
    


    
![png](output_32_1.png)
    



```python
# calling a function for an image prediction

def predict(model , image):
    img = tf.keras.preprocessing.image.img_to_array(image)
    reshape = tf.expand_dims(img, 0)

    predict = model.predict(reshape)

    print(predict)
```


```python
for image_32_batch, label_32_batch in testing_set.take(1):
    print(image_32_batch[0].numpy().shape)
    predict(model, image_32_batch[0])
```

    (256, 256, 3)
    1/1 [==============================] - 0s 363ms/step
    [[0.00427388]]
    


```python
predictions = model.predict(testing_set)

predicted_labels = np.argmax(predictions, axis=1)

true_labels = np.concatenate([y for x, y in testing_set], axis=0)

print(classification_report(true_labels, predicted_labels, target_names=['Closed_Eyes', 'Open_Eyes']))

```

    65/65 [==============================] - 2s 26ms/step
                  precision    recall  f1-score   support
    
     Closed_Eyes       0.49      1.00      0.66      1029
       Open_Eyes       0.00      0.00      0.00      1051
    
        accuracy                           0.49      2080
       macro avg       0.25      0.50      0.33      2080
    weighted avg       0.24      0.49      0.33      2080
    
    

    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python
predictions = model_mobile.predict(testing_set)

predicted_labels = np.argmax(predictions, axis=1)

true_labels = np.concatenate([y for x, y in testing_set], axis=0)

print(classification_report(true_labels, predicted_labels, target_names=['Closed_Eyes', 'Open_Eyes']))
```

    65/65 [==============================] - 6s 63ms/step
                  precision    recall  f1-score   support
    
     Closed_Eyes       0.49      1.00      0.66      1029
       Open_Eyes       0.00      0.00      0.00      1051
    
        accuracy                           0.49      2080
       macro avg       0.25      0.50      0.33      2080
    weighted avg       0.24      0.49      0.33      2080
    
    

    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python

```
