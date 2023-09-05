import torch
import categories as cat

# load model
classes = cat.category
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# load images
PATH = 'data/'
suffix = '.jpg'
imgs = []
for i in range(1,14):
    imgs.append(PATH+str(i)+suffix)

for img in imgs:
    # model prediction
    results = model(img)

    # print result in console
    result = results.xyxy[0].numpy()
    i = 0
    for item in result:
        print(f'item {i}: {classes[int(item[5])]}')
        print(f'coordinate: from [{int(item[0])},{int(item[1])}] to [{int(item[2])},{int(item[3])}]')
        print(f'confidence: {round(item[4]*100,2)}%\n')
        i += 1
    results.show()
    c = input("Proceed to next image? (y/n)\n")
    if c == 'n':
        break

