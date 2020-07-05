# Computer-Pointer-Controller-OPENVINO
Using this python application you can control computer pointer with your eye gaze. This project uses four pretrained neural network models, which includes eye gaze estimation model too (downloaded from Intel's OPENVINO toolkit.
##Terminal command used:
python3 /home/workspace/CPC_project/src/main.py -t video -i /home/workspace/CPC_project/bin/demo.mp4 -f /home/workspace/CPC_project/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -p /home/workspace/CPC_project/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 -l /home/workspace/CPC_project/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 -g /home/workspace/CPC_project/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 -FDO 1 -FLD 1

![FPS vs Models precisions](./results_testing/fps_avg_sync_vs_async.jpeg)
