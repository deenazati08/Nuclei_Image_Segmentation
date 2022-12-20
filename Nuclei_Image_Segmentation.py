# %%
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os, cv2, glob, datetime

# %%
# 1. Data Loading
# Train Data
TRAIN_PATH = os.path.join(os.getcwd(), 'dataset', 'train')
images = []
masks = []

# A. Load images
image_dir = os.path.join(TRAIN_PATH, 'inputs')
for image_file in os.listdir(image_dir) :
    img = cv2.imread(os.path.join(image_dir, image_file))       # Read the image file based on the full path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                  # Convert the image from bgr to rgb
    img = cv2.resize(img, (128,128))                            # Resize the image into 128x128
    images.append(img)                                          # Place the image into the empty list

# B. Load masks
mask_dir = os.path.join(TRAIN_PATH, 'masks')
for mask_file in os.listdir(mask_dir) :
    mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128,128))
    masks.append(mask)                           

# Test Data
TEST_PATH = os.path.join(os.getcwd(), 'dataset', 'test')
test_images = []
test_masks = []

# A. Load images
test_image_dir = os.path.join(TEST_PATH, 'inputs')
for test_image_file in os.listdir(test_image_dir) :
    test_img = cv2.imread(os.path.join(test_image_dir, test_image_file))       
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)                  
    test_img = cv2.resize(test_img, (128,128))                            
    test_images.append(test_img)                                          

# B. Load masks
test_mask_dir = os.path.join(TEST_PATH, 'masks')
for test_mask_file in os.listdir(test_mask_dir) :
    test_mask = cv2.imread(os.path.join(test_mask_dir, test_mask_file), cv2.IMREAD_GRAYSCALE)
    test_mask = cv2.resize(test_mask, (128,128))
    test_masks.append(test_mask)                           

# %%
# 2. Data Preparation
# Convert list to np array
images_np = np.array(images)
masks_np = np.array(masks)

test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

# Check some examples
plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(images_np[i])
    plt.axis('off')

print('Example of train images')    
plt.show()

plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(masks_np[i])
    plt.axis('off')

print('Example of train masks')        
plt.show()

plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(test_images_np[i])
    plt.axis('off')
    
print('Example of test images')    
plt.show()

plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(test_masks_np[i])
    plt.axis('off')
    
print('Example of test masks')        
plt.show()

# %%
# Expand the mask dimension to include the channel axis
masks_np_exp = np.expand_dims(masks_np, axis=-1)
test_masks_np_exp = np.expand_dims(test_masks_np, axis=-1)
# Check the mask output
print(np.unique(masks[0]))      
print(np.unique(test_masks[0]))

# %%
# Normalize the images pixel values
converted_images = images_np/255.0
test_converted_images = test_images_np/255.0

converted_masks = (masks_np_exp > 128)*1
test_converted_masks = (test_masks_np_exp > 128)*1

# %%
# Perform train test split
SEED = 12345
X_train, X_test, y_train, y_test = train_test_split(converted_images, converted_masks,random_state=SEED)

# %%
# Convert the numpy array into tensorflow tensors
X_train_tensor = tf.data.Dataset.from_tensor_slices(X_train)
X_test_tensor = tf.data.Dataset.from_tensor_slices(X_test)
X_testtest_tensor = tf.data.Dataset.from_tensor_slices(test_converted_images)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)
y_testtest_tensor = tf.data.Dataset.from_tensor_slices(test_converted_masks)

# Combine features and labels together to form a zip dataset
train = tf.data.Dataset.zip((X_train_tensor, y_train_tensor))
test = tf.data.Dataset.zip((X_test_tensor, y_test_tensor))
testtest = tf.data.Dataset.zip((X_testtest_tensor, y_testtest_tensor))

# %%
# Create a subclass layer for data augmentation
class Augment(layers.Layer):
    def __init__(self, seed=SEED):
        super().__init__()
        self.augment_inputs = layers.RandomFlip(mode='horizontal', seed=seed)
        self.augment_labels = layers.RandomFlip(mode='horizontal', seed=seed)
        
    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs,labels
   
# %%
#2.7. Convert into prefetch dataset
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train)
STEPS_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE

train_batches = (
    train
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

test_batches = test.batch(BATCH_SIZE)
testtest_batches = testtest.batch(BATCH_SIZE)

# %%
# Visualize some pictures as example
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    
    for i in range (len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()

# %%
for images, masks in train_batches.take(1):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])

# %%
# 3. Model Development
# Create image segmentation model
# Use a pretrained model as the feature extraction layers
base_model = MobileNetV2(input_shape=[128,128,3], include_top=False)

# List down some activation layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Define the feature extraction model
down_stack = Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

# Define the upsampling path
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = layers.Input(shape=[128,128,3])
    # Apply functional API to construct U-Net
    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    # Upsampling and establishing the skip connections(concatenation)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = layers.Concatenate()
        x = concat([x, skip])
        
    # This is the last layer of the model (output layer)
    last = layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2, padding='same') #64x64 --> 128x128
    
    x = last(x)
    
    return Model(inputs=inputs, outputs=x)

# %%
# Make of use of the function to construct the entire U-Net
OUTPUT_CLASSES = 2
model = unet_model(output_channels=OUTPUT_CLASSES)

# Compile the model
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
plot_model(model, show_shapes=True)

# %%
# Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
            
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])

# %%
# Test out the show_prediction function
show_predictions()

# %%
# Create a callback to help display results during model training
class DisplayCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))
   
LOG_DIR = os.path.join(os.getcwd(),'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = TensorBoard(log_dir=LOG_DIR)
es = EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True)

# %%
# Hyperparameters for the model
EPOCHS = 15
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(test)//BATCH_SIZE//VAL_SUBSPLITS

history = model.fit(train_batches, validation_data=test_batches, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, callbacks=[DisplayCallback(), tb, es])

# %%
# 4. Model Deployment
show_predictions(test_batches,3)

# Evaluate the model
print(model.evaluate(testtest_batches))

# %%
# Save model
model.save('model.h5')
