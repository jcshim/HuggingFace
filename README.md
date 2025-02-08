# HuggingFace

예제는 OpenCV의 Haar Cascade를 사용해서 USB 카메라로 얼굴을 검출하고, 검출된 얼굴 주위에 사각형 박스를 그리는 코드야.  
Hugging Face는 정말 멋진 플랫폼이지만, 얼굴 검출의 경우에는 OpenCV에서 제공하는 Haar Cascade가 간단하고 실행하기도 편해서 이번 실습에서는 그걸 사용해봤어.  
(참고로, 더 정교한 얼굴 검출을 원한다면 Hugging Face Hub에서 MTCNN 같은 딥러닝 기반 모델을 찾아볼 수도 있어. 예를 들어, [facenet-pytorch](https://github.com/timesler/facenet-pytorch) 라이브러리도 좋은 선택이야.)

먼저, 아래 패키지들을 설치해야 해:  
```bash
pip install opencv-python
```

그리고 아래의 파이썬 코드를 실행해봐. USB 카메라(보통은 0번 디바이스)를 열어서 얼굴을 실시간으로 검출하고 화면에 표시해 줄 거야. (창을 닫으려면 **q** 키를 누르면 돼.)

```python
import cv2

# Haar Cascade XML 파일 경로 (OpenCV 내장 경로 사용)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

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

    # 얼굴 검출을 위해 그레이스케일 이미지로 변환 (Haar Cascade는 그레이스케일 이미지를 사용)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출 (scaleFactor와 minNeighbors 값은 상황에 맞게 조절 가능)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 검출된 얼굴마다 사각형 박스 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 결과 영상 출력
    cv2.imshow("Face Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**코드 설명**  
- **cv2.data.haarcascades**: OpenCV에 기본 내장되어 있는 Haar Cascade XML 파일들이 저장된 경로야. 여기서 얼굴 검출용 XML 파일을 불러와.
- **VideoCapture(0)**: 기본 USB 카메라를 열어. 만약 외부 카메라가 여러 개 연결되어 있다면, 번호를 조정해줘.
- **detectMultiScale**: 그레이스케일 이미지에서 얼굴을 검출해. `scaleFactor`와 `minNeighbors` 값은 검출 성능에 영향을 주니, 필요에 따라 조정하면 좋아.
- **cv2.rectangle**: 검출된 얼굴 주위에 사각형 박스를 그려.

출처는 [OpenCV 공식 문서](https://docs.opencv.org/)를 참고했어.

이 코드 실행 후에 카메라가 켜지고, 얼굴이 감지되면 사각형이 표시되는 걸 볼 수 있을 거야. 실행 중 창이 뜨면, **q** 키를 눌러서 종료하면 돼.
