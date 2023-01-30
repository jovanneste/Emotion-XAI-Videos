from lime_video import *
from mask import *
import sys

print("Loading model...")
global model
model = keras.models.load_model('../../data/models/predict_model')
print(model.summary())
#
# video_path = "../../data/videos/train_videos/421.mp4"
# prime_frame, lower_frame, upper_frame, frameSize, fps = maskFrames(video_path, 15)
#
# print('\n\n\n')
# print(prime_frame, lower_frame, upper_frame, frameSize, fps)
#this returns 20 12 24 (640, 464) 29.97
createMaskedVideos(20, 12, 24-1, 29.97, (640,464), 5)

file = open('segments_and_prime_frame', 'rb')
segments_and_prime_frame = pickle.load(file)
segments = segments_and_prime_frame[0]
prime_frame_num = segments_and_prime_frame[1]
file.close()

originl_video = '../../data/LIMEset/0.mp4'
prime_frame_img = cv2.imread('../../data/frames/frame'+str(prime_frame_num)+'.jpg')
explainer = LimeVideoExplainer()
explanation = explainer.explain_instances(originl_video, model.predict, segments)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], prime_frame_num, num_features=5, hide_rest=False)

plt.imshow(mask)
plt.show()
plt.imshow(prime_frame_img)
plt.show()


plt.imshow(mark_boundaries(prime_frame_img, mask))
plt.show()
