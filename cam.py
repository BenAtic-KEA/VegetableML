import cv2
from fastai.vision import *
from fastai.vision.all import *
from fastai.basics import *

learn = load_learner('vegetables_model.pkl',cpu=True)
vid = cv2.VideoCapture(0)

while (True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    
        
    t = torch.tensor(np.ascontiguousarray(np.flip(frame, 2)).transpose(2,0,1)).float()/255
    img = Image(t) # fastai.vision.Image, not PIL.Image
    p = learn.predict(img)
# If needed, convert the frame to grayscale
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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