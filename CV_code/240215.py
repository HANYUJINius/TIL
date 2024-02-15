import sys
import os
# darknet 라이브러리 설정
sys.path.append(os.path.join(os.getcwd(), '/home/hyj/opencv/opencv-4.4.0/build/darknet/python'))
import cv2
import darknet as dn

# darknet 라이브러리 설정
dn.set_gpu(0)
net = dn.load_net(b"/hyj/ddwu/opencv/opencv-4.4.0/build/darknet/cfg/yolov3.cfg", b"/home/hyj/opencv/opencv-4.4.0/build/darknet/yolov3.weights", 0)
meta = dn.load_meta(b"/hyj/ddwu/opencv/opencv-4.4.0/build/darknet/cfg/coco.data")

# OpenCV 설정
cap = cv2.VideoCapture(0, cv2.CAP_V4L)

if cap.isOpened():
    file_path = 'out.mp4'
    fps = 25.40
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 인코딩 포맷 문자
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)  # 프레임 크기

    out = cv2.VideoWriter(file_path, fourcc, fps, size)  # VideoWriter 객체 생성

    while True:
        ret, frame = cap.read()

        if ret:
            # 코드 추가
            r = dn.detect(net, meta, frame)

            for detection in r:
                label = detection[0]
                confidence = detection[1]
                bbox = detection[2]

                left, top, right, bottom = bbox
                left, top, right, bottom = int(left), int(top), int(right), int(bottom)

                # 사각형 그리기
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # 레이블과 신뢰도 표시
                label_text = f"{label.decode('utf-8')}: {confidence:.2f}"
                cv2.putText(frame, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # # 기존 코드
            cv2.imshow('demo', frame)
            out.write(frame)  # 파일 저장

            if cv2.waitKey(int(1000/fps)) != -1:
                break
        else:
            print(b'no file!')
            break

    out.release()  # 파일 닫기
else:
    print(b"Can't open camera!")

cap.release()
cv2.destroyAllWindows()
