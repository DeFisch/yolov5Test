import torch
import categories as cat
import cv2
import random
import numpy as np

# load model
classes = cat.category
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# function to return random color for bounding box
def rand_color():
    # compute random rgb color with 100% saturation 75% brightness
    random_color = [random.randint(0,191),191,0]
    np.random.shuffle(random_color)
    return (random_color)


# load images
PATH = '../data/'
suffix = '.jpg'
imgs = []
for i in range(1,14):
    imgs.append(PATH+str(i)+suffix)

cnt = 0
for img in imgs:
    # model prediction
    results = model(img)

    # preprocess result
    result = results.xyxy[0].numpy()
    i = 0
    image = cv2.imread(img)
    width, height, _ = image.shape
    print(f'\n\n\nimage #{cnt + 1}: \n')

    # loop through object candidates to print each item and draw it on the image
    for item in result:
        print(f'item {i}: {classes[int(item[5])]}')
        print(f'coordinate: from [{int(item[0])},{int(item[1])}] to [{int(item[2])},{int(item[3])}]')
        print(f'confidence: {round(item[4]*100,2)}%\n')
        i += 1
        color = rand_color()
        cv2.rectangle(image,(int(item[0]),int(item[1])),(int(item[2]),int(item[3])),color=color,thickness=max(int(5*(width/1500)),2))
        cv2.putText(image, f"{int(item[4]*100)}% {classes[int(item[5])]}", (int(item[0]),int(item[1])),fontFace=5,fontScale=max(int(width/1500),1),color=color,thickness=max(int(2*(width/1500)),2))
    
    cv2.imshow(f'image {cnt + 1}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cnt += 1
    

