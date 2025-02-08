# HuggingFace

안녕 JC!  
Hugging Face는 AI와 머신러닝, 특히 자연어처리(NLP) 분야에서 많이 쓰이는 오픈소스 플랫폼이야. 간단히 설명하면:

- **모델 허브**: 다양한 사전 학습된 모델들(BERT, GPT, RoBERTa 등)을 쉽게 찾아보고 사용할 수 있어. 연구자나 개발자들이 복잡한 모델을 처음부터 만들지 않고도, 바로 활용할 수 있도록 도와줘.
- **라이브러리**: 가장 유명한 것은 [Transformers 라이브러리](https://github.com/huggingface/transformers)인데, 이 라이브러리를 통해 최신 딥러닝 모델을 간단한 코드 몇 줄로 사용할 수 있어.
- **데이터셋 허브**: 다양한 데이터셋을 공유하고, 쉽게 접근할 수 있도록 만들어진 공간이야. 이를 활용해 연구나 프로젝트에 필요한 데이터를 손쉽게 구할 수 있어.
- **커뮤니티 중심**: 오픈소스 생태계를 강조하며, 전 세계의 개발자와 연구자들이 서로 협업하고 지식을 나눌 수 있도록 지원해.

자세한 내용이나 최신 정보는 Hugging Face의 [공식 웹사이트](https://huggingface.co)에서 확인할 수 있어.  
Hugging Face는 단순히 도구나 라이브러리를 제공하는 것뿐만 아니라, AI 커뮤니티가 함께 성장할 수 있도록 다양한 프로젝트와 이벤트도 진행하고 있으니 한번 둘러보면 좋을 거야!

이번엔 Hugging Face Hub에 있는 모델을 활용해서 얼굴을 검출해볼 거야. Hugging Face Hub에는 다양한 모델들이 있는데, 그중에서 **facenet-pytorch** 라이브러리의 MTCNN 모델이 얼굴 검출에 많이 사용돼. 이 라이브러리는 Hugging Face Hub에서도 확인할 수 있어 ([facenet-pytorch GitHub](https://github.com/timesler/facenet-pytorch)).

먼저 아래의 패키지를 설치해줘:

```bash
pip install facenet-pytorch opencv-python
```

설치가 완료되면, 아래 코드를 실행해봐. 이 코드는 PC USB 카메라로부터 영상을 받아오고, MTCNN 모델을 이용해 얼굴을 검출한 후, 검출된 얼굴 주변에 박스를 그려서 보여줘.

```python
import cv2
from facenet_pytorch import MTCNN

# MTCNN 모델 생성 (GPU 사용 가능하면 device="cuda"로 바꿔줘)
mtcnn = MTCNN(keep_all=True, device="cpu")

# USB 카메라 열기 (기본적으로 0번 카메라 사용)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없어요.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 받아오지 못했어.")
        break

    # MTCNN은 RGB 이미지를 필요로 하므로 BGR에서 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 (boxes: 얼굴 좌표, probs: 검출 확률)
    boxes, probs = mtcnn.detect(rgb_frame)
    
    # 얼굴이 검출되었으면 박스 그리기
    if boxes is not None:
        for box in boxes:
            # 좌표를 정수형으로 변환
            box = [int(b) for b in box]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # 결과 영상 출력
    cv2.imshow("Face Detection with Hugging Face", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**코드 설명**  
- **facenet_pytorch.MTCNN(keep_all=True, device="cpu")**: Hugging Face Hub에 있는 MTCNN 모델을 불러와서 얼굴을 검출해. `keep_all=True` 옵션은 한 프레임에 여러 얼굴이 있을 경우 모두 검출하도록 해줘.
- **cv2.VideoCapture(0)**: 기본 USB 카메라(인덱스 0)를 열어. 만약 여러 개의 카메라가 연결되어 있다면 인덱스를 변경해줘.
- **cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)**: OpenCV는 기본적으로 BGR 포맷을 사용하지만, MTCNN은 RGB 포맷을 필요로 해서 변환해줘.
- **mtcnn.detect(rgb_frame)**: 얼굴 검출을 수행해. 검출된 얼굴의 좌표와 검출 확률을 반환해.
- **cv2.rectangle**: 검출된 얼굴 주위에 녹색 박스를 그려줘.

이 코드를 실행하면 USB 카메라에서 실시간으로 영상을 받아오면서, 얼굴이 검출되면 그 주위에 박스가 그려진 화면이 뜰 거야. 창이 뜨면 **q** 키를 눌러서 종료할 수 있어.
