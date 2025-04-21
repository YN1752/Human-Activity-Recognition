import glob
import cv2
import time
import torch
from torchvision import transforms
from model import CNNLSTM
from autotagging import Tags
import sys
import warnings

warnings.filterwarnings('ignore')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNLSTM(num_classes=50)
model.load_state_dict(torch.load("model.pth", weights_only=True))

labels = [action.split("\\")[1] for action in  glob.glob("UCF50/*")]


def extract_frames(path):
    cap = cv2.VideoCapture(path)
    interval = max(cap.get(cv2.CAP_PROP_FRAME_COUNT)//20, 1)
    frames_list = []

    for i in range(20):
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, i*interval)
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames_list.append(frame)

    cap.release()

    return torch.stack(frames_list)


def display_prediction(path, predicted):
    cap = cv2.VideoCapture(path)
    fps = 60

    while True:
        ret, video = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', 800, 600)
        
        video = cv2.putText(video, "Activity: " + predicted, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Video",video)

        time.sleep(1 / fps)

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def predict_activity(path):
    frames = extract_frames(path).unsqueeze(0)
    input = frames.to(device)

    model.to(device)

    model.eval()
    with torch.no_grad():
        softmax = torch.softmax(model(input), dim=1)
        predicted = labels[torch.argmax(softmax).item()]
    tags = Tags.activity_tags[predicted]

    print("Tags:")
    for tag in tags:
        print("#"+tag)

    display_prediction(path, predicted)


if __name__ == "__main__":
    predict_activity("test/"+sys.argv[1])
    