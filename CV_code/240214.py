import sys, os
# darknet 라이브러리 설정
sys.path.append(os.path.join(os.getcwd(), '/home/hyj/opencv/opencv-4.4.0/build/darknet/python'))

import cv2
import darknet as dn
import pdb

# darknet 라이브러리 설정
sys.path.append(os.path.join(os.getcwd(), '/home/hyj/opencv/opencv-4.4.0/build/darknet'))
dn.set_gpu(0)
net = dn.load_net(b"/home/hyj/opencv/opencv-4.4.0/build/darknet/cfg/yolov3.cfg", b"/home/ddwu/opencv/opencv-4.4.0/build/darknet/yolov3.weights", 0)
meta = dn.load_meta(b"/home/hyj/opencv/opencv-4.4.0/build/darknet/cfg/coco.data")

# OpenCV 설정
cap = cv2.VideoCapture(0,cv2.CAP_V4L)

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
        	r = dn.detect(net, meta, frame)
        	cv2.imshow('demo', frame)
        	out.write(frame)                            # 파일 저장
        	if cv2.waitKey(int(1000/fps)) != -1:
        		break
        else:
        	print(b'no file!')
        	break
        	out.release()                                       # 파일 닫기
else:
    print(b"Can`t open camera!")
cap.release()
cv2.destroyAllWindows()
