
# %%
import os
import torch
from IPython.display import Image, clear_output 
print('PyTorch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
# %%
#Next we need to use terminal coomands:
'''
!git clone https://github.com/roboflow-ai/yolov3  # clone
!curl -L "https://github.com/yardet/Transfer_learning_yolov3" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
%cd train
%ls
%mkdir labels
%mkdir images
%mv *.jpg ./images/
%mv *.txt ./labels/
%cd images 
'''
# %%
# create Ultralytics specific text file of training images
file = open("train_images_roboflow.txt", "w") 
for root, dirs, files in os.walk("."):
    for filename in files:
      # print("../train/images/" + filename)
      if filename == "train_images_roboflow.txt":
        pass
      else:
        file.write("../train/images/" + filename + "\n")
file.close()
# %%
#More terminal commands:
'''
%cat train_images_roboflow.txt
%cd ../../valid
%mkdir labels
%mkdir images
%mv *.jpg ./images/
%mv *.txt ./labels/
%cd images
'''
#%%
# create Ultralytics specific text file of validation images
file = open("valid_images_roboflow.txt", "w") 
for root, dirs, files in os.walk("."):
    for filename in files:
      # print("../train/images/" + filename)
      if filename == "valid_images_roboflow.txt":
        pass
      else:
        file.write("../valid/images/" + filename + "\n")
file.close()
#%%
# update the roboflow.data file with correct number of classes
import re

num_classes = get_num_classes("../../train/_darknet.labels")#
with open("roboflow.data") as f:
    s = f.read()
with open("roboflow.data", 'w') as f:
    
    # Set number of classes num_classes.
    s = re.sub('classes=[0-9]+',
               'classes={}'.format(num_classes), s)
    f.write(s)

#%%
# Now we are ready to train the model: use this for tarning
'''
%cd /content/yolov3
!python3 train.py --data data/roboflow.data --epochs 80
'''
#after traning make the weights
'''
!python3 detect.py --weights weights/last.pt --source=../test --names=../train/roboflow_data.names
'''
#now the model is ready for work
#%%
## Run the model on our video
'''open dir for frames:
%cd /content/
%mkdir framess
%cd /content/framess
'''
cap = cv2.VideoCapture("/content/RaccoonVideo3.mp4")

if (cap.isOpened() == False):
  print("Unable to read video")

ret=True
i=0
while(ret):
  ret, frame = cap.read()
 
  if ret == True:
    i+=1
    name="g"+str(i)+".jpg"
    cv2.imwrite(name , frame)
#%%
#now run the model on all frames
'''
!python3 detect.py --weights weights/last.pt --source=/content/framess --names=../train/roboflow_data.names
'''
#%%now we need to stich the results into a video
# first go to output dir : %cd/content/yolov3/output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/content/framess/RaconOut4.mp4', fourcc, 30, (int(cap.get(3)),int(cap.get(4))))
i=1

while(i<408):
  name="g"+str(i)+".jpg"
  out.write(cv2.imread(name))
  i+=1
out.release()
model.load_state_dict(chkpt['model'], strict=False)