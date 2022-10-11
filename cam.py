import cv2
import sys
sys.version
from fastai.imports import *
from fastai.vision.all import *
from fastai.data.all import *
import fastbook
fastbook.setup_book()
from fastai.callback.preds import get_image_files
from fastai.vision.all import CategoryBlock
from duckduckgo_search import ddg_images
from fastcore.all import *
from fastdownload import download_url

learn = load_learner('vegetables_model.pkl',cpu=True)
vid = cv2.VideoCapture(0)

while (True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    if ret==True:
        
        is_vege,_,probs = learn.predict(PILImage.create(frame))
    # If needed, convert the frame to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.putText(frame, f"{is_vege}", (10,100),
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