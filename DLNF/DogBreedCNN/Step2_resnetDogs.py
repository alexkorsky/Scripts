import random
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import cv2  
import matplotlib.pyplot as plt 
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))

human_files_short = human_files[:100]
dog_files_short = train_files[:100]

print("Loaded dog and human files\n")

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
print("Loaded ResNet\n")


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
    


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
    
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))     
    

# What percentage of the images in human_files_short have a detected dog?
# What percentage of the images in dog_files_short have a detected dog?

### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
#tensorList = dog_files_short #paths_to_tensor(dog_files_short);
#for imgTensor in tensorList:
#    isDog = dog_detector(imgTensor)
#    print("FIle: ", imgTensor)
#    print("IsDOg: ", isDog)


truePositive = 0
falseNegative = 0
for fileName in dog_files_short:
    isDog = dog_detector(fileName)
    if (isDog):
        truePositive +=1
    else:
        falseNegative += 1
        
        '''
        img = cv2.imread(fileName)
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # display the image, along with bounding box
        plt.imshow(cv_rgb)
        plt.show()
        '''
        
        
print("True Positive: %d" % truePositive)
print("False Negative: %d" % falseNegative)
      
trueNegative = 0
falsePositive = 0
for fileName in human_files_short:
    isDog = dog_detector(fileName) 
    if (isDog):
        falsePositive +=1
        
        '''
        img = cv2.imread(fileName)    
        # convert BGR image to RGB for plotting
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       
        # display the image, along with bounding box
        plt.imshow(cv_rgb)
        plt.show()
        '''
    else:
        trueNegative += 1
        
print("True Negative: %d" % trueNegative)
print("False Positive: %d" %  falsePositive)


print("--------------")
total = truePositive + falseNegative + trueNegative + falsePositive
print("Accuracy: %.2f" % ((truePositive + trueNegative) / total))
print("Precision: %.2f" % (truePositive / (truePositive + falsePositive)))
print("Recall: %.2f" % (truePositive / (truePositive + falseNegative)))




"""
# Save Data
with open('resnet50.pickle', 'wb') as out_file:
    pickle.dump(ResNet50_model, out_file)
    
# Reload the data
pickle_file = 'resnet50.pickle'
with open(pickle_file, 'rb') as f:
  ResNet50_model2 = pickle.load(f)
  
print (ResNet50_model)
print("XXX")
print (ResNet50_model)
"""