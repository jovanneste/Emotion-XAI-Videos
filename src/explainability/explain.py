from lime_video import *
from mask import *


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
prime_frame = segments_and_prime_frame[1]
file.close()

originl_video = '../../data/LIMEset/0.mp4'
explainer = LimeVideoExplainer()
explanation = explainer.explain_instances(originl_video, model.predict, segments)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], prime_frame, num_features=10, hide_rest=False)
print(temp.shape)
print(temp)
print(mask.shape)
print(mask)

plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

cv2.imshow("Mask", mask)
cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
