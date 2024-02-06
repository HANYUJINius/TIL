import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened:                                        #카메라가 열렸는지 확인
    file_path = './out.mp4'                             #경로 지정
    fps = 25.40                                         #출력 비디오의초당 프레임 수 설정
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')            #인코딩 포맷 문자. 압축을 위해 사용되는 비디오 코덱 지정. DIVX코덱 사용
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)           #프레임의 너비
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)         #프레임의 높이
    size = (int(width), int (height))                   #프레임 크기
    
    out = cv2.VideoWriter(file_path, fourcc, fps, size) #VideoWriter 객체('out') 생성
    while True:
        ret, frame = cap.read()                         #카메라에서 프레임 읽기, 'ret'에는 성공 여부를 나타내는 T/F, "frame"에는 캡쳐된 프레임 저장
        if ret:
            cv2.imshow('camera-recording', frame)       #캡쳐된 프레임을 'camera-recording'이라는 창에 표시
            out.write(frame)                            #파일 저장
            if cv2.waitKey(int(1000/fps)) != -1:        #fps에 기반해 짧은 시간동안 일시 중지. 이 기간 동안 키가 눌리면 루프 종료
                break
        else:
            print('no file!')
            break
    out.release()                                       #리소스 해제 및 창 닫기
else:
    print("Can`t open camera!")
cap.release()                                           #VideoWriter 객체를 해제
cv2.destroyAllWindows()                                 #출력 비디오 파일 닫기