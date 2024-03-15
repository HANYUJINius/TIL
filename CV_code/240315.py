import logging      #로깅 모듈
import mimetypes    #mime 타입 처리를 위한 모듈
import os           #파일 및 경로 관리를 위한 모듈
import time         #시간 관리를 위한 모듈
from argparse import ArgumentParser #명령줄 인자 파싱을 위한 모듈
from argparse import Namespace

import cv2                      #OpenCV라이브러리
import json_tricks as json      #JSON 데이터 처리를 위한 모듈
import mmcv                     #MMComputerVision 라이브러리
import mmengine                 #MMEngine 라이브러리
import numpy as np              #numpy
from mmengine.logging import print_log  #로깅 함수

from mmpose.apis import inference_topdown       #mmpose에서 상체 포즈 추론 API
from mmpose.apis import init_model as init_pose_estimator   #포즈 모델 초기화 API
from mmpose.evaluation.functional import nms                #NMS(non-maximum suppression)함수
from mmpose.registry import VISUALIZERS                     #시각화 모듈 registry 
from mmpose.structures import merge_data_samples, split_instances #데이터 샘플 처리 함수
from mmpose.utils import adapt_mmdet_pipeline                     #MMDetection pipeline 수정 함수

try:
    from mmdet.apis import inference_detector, init_detector    #MMDection API
    has_mmdet = True        #MMDetection이 설치되어 있는지 확인하기 위한 변수
except (ImportError, ModuleNotFoundError):     
    has_mmdet = False       #MMDetection이 설치되어 있지 않음을 타나내는 변수

def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)

def main():
    args = Namespace(
    det_config='../demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
    det_checkpoint='../checkpoint/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
    pose_config='../configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
    pose_checkpoint='../checkpoint/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth',
    input='../inputimg/17.jpg',
    show= True,
    output_root='./ouputvideo',
    save_predictions=True,
    device='cuda:0',
    det_cat_id=0,
    bbox_thr=0.3,
    nms_thr=0.3,
    kpt_thr=0.3,
    draw_heatmap=True,
    show_kpt_idx=True,
    skeleton_style='mmpose',
    radius=3,
    thickness=1,
    show_interval=0,
    alpha=0.8,
    draw_bbox=True
    )

    assert args.show or (args.output_root != '')    #이미지를 보여줄지 또는 시각화 이미지를 저장할 디렉토리가 있어야 함
    assert args.input != ''                         #입력 파일이 있어야 함
    assert args.det_config is not None              #검출을 위한 설정 파일이 있어야 함
    assert args.det_checkpoint is not None          #검출을 위한 checkpoint file이 있어야 함

    output_file = '/home/ddwu/mmpose/outputvideo/result.mp4'    #출력 파일 경로 설정
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)   #출력 경로가 지정되어 있으면 디렉토리 생성
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'   #웹캠으로부터 입력을 받을 경우 MP4 확장자 추가

    if args.save_predictions:
        assert args.output_root != ''   #결과를 저장할 경로가 지정되어 있어야 함
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json' #예측 결과 저장 경로 설정

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)   #detection model 초기화
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)   #MMDetection pipeline 수정

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))  #pose model 초기화

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius  #시각화를 위한 keypoint 반지름 설정
    pose_estimator.cfg.visualizer.alpha = args.alpha    #bbox 투명도 설정
    pose_estimator.cfg.visualizer.line_width = args.thickness   #시각화를 위한 링크 두께 설정
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)   #시각화 모듈 빌드
    # the dataset_meta is loaded from the checkpoint and then pass to the model in init_pose_estimator
    # 데이터셋 메타 정보를 불러와 모델에 전달
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]  #입력 파일의 MIME 타입 추론

    if input_type == 'image':   #이미지 파일이면

        # inference 추론 수행
        pred_instances = process_one_image(args, args.input, detector,
                                           pose_estimator, visualizer)  #이미지 처리

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)   #예측 결과 분할

        if output_file:
            img_vis = visualizer.get_image()    #시각화된 이미지 가져오기
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)    #이미지 저장

    elif input_type in ['webcam', 'video']: #웹캠 또는 비디오 파일이면

        if args.input == 'webcam':
            cap = cv2.VideoCapture(0)   #웹캠 사용
        else:
            cap = cv2.VideoCapture(args.input)  #비디오 파일 사용

        video_writer = None #비디오 작성기 초기화
        pred_instances_list = []    #예측 결과 리스트 초기화
        frame_idx = 0   #프레임 인덱스 초기화

        while cap.isOpened():   #비디오가 열려 있는 동안 반복
            success, frame = cap.read() #비디오 프레임 읽기
            frame_idx += 1  #프레임 인덱스 증가

            if not success:
                break   #비디오가 끝나면 반복 종료

            # topdown pose estimation 
            pred_instances = process_one_image(args, frame, detector,
                                               pose_estimator, visualizer,
                                               0.001)   #프레임 처리

            if args.save_predictions:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances))) #예측 결과 저장

            # output videos 비디오 출력
            if output_file:
                frame_vis = visualizer.get_image()  #시각회된 이미지 가져오기

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    #초기화
                    # the size of the image with visualization may vary depending on the presence of heatmaps
                    # 시각화된 이미지 크기는 히트맵 유무에 따라 달라짐
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        30,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis)) #비디오 프레임 저장

            if args.show:
                # press ESC to exit
                # ESC 키를 눌러 종료
                if cv2.waitKey(5) & 0xFF == 27: #ESC 키를 누르면 종료
                    break

                time.sleep(args.show_interval)  #대기 시간 설정

        if video_writer:
            video_writer.release()  #해제

        cap.release()   #비디오 캡쳐 객체 해제

    else:
        args.save_predictions = False   #예측 결과 저장 비활성화
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.') #잘못된 형식의 파일 예외 처리

    if args.save_predictions:
        with open(args.pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')    #예측 결과 저장
        print(f'predictions have been saved at {args.pred_save_path}')  #예측 결과 저장 메시지 출력

    if output_file:
        input_type = input_type.replace('webcam', 'video')  #웹캠일 경우 비디오로 변경
        print_log(
            f'the output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO) #출력 파일 저장 메시지를 출력


if __name__ == '__main__':
    main()  #메인 함수 호출
