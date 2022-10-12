from typing import final
import cv2
from PIL import Image
from fastai.vision.core import *
from fastai.vision.widgets import *
from fastai.vision.all import *
from fastai.basics import *
from fastai.data.external import *
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image

learn = load_learner('vegetables_model1.pkl')
vid = cv2.VideoCapture(0)
transform = T.ToTensor()
while (True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    
    img_read = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_nump = np.asarray(img_read)
    tensor_img = transform(img_read)
    img = Image.fromarray((img_nump*255).astype(np.uint8))
    print(img)
    ##print(img_tensor)
    #img_pil = Image.fromarray(img_read)
    p,_,pred = learn.predict(TensorImageBase(img))

    cv2.putText(frame, f"{p}", (10,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,(255,255,255), 4, 2)

# Display the resulting frame
    cv2.imshow('Camera feed', frame)

# the 'q' button is set as the
# quitting button you may use any
# desired button of your choice

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()