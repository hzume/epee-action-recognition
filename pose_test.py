import cv2

from mmpose.apis import MMPoseInferencer

img_path = "input/data/frames/2025-01-04_08-40-12_0.jpg"
inferencer = MMPoseInferencer(pose2d="human")

result_generator = inferencer(img_path, return_vis=True)
result = next(result_generator)
print(result)

img = result["visualization"][0]
cv2.imwrite("pose_test.png", img)
