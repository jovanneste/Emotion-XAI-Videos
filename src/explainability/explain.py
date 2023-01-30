# from lime_video import *
# from mask import *
import sys
import argparse


def explain_model_prediction(video_path, model):
    prime_frame, lower_frame, upper_frame, frameSize, fps = maskFrames(video_path, 15)

    print(prime_frame, lower_frame, upper_frame, frameSize, fps)

    print("Creating masks...")
    createMaskedVideos(prime_frame, lower_frame, upper_frame-1, fps, frameSize, 200)

    file = open('segments_and_prime_frame', 'rb')
    segments_and_prime_frame = pickle.load(file)
    segments = segments_and_prime_frame[0]
    prime_frame_num = segments_and_prime_frame[1]
    file.close()

    originl_video = '../../data/LIMEset/0.mp4'
    prime_frame_img = cv2.imread('../../data/frames/frame'+str(prime_frame_num)+'.jpg')

    print("Creating LIME explainer...")
    explainer = LimeVideoExplainer()
    explanation = explainer.explain_instances(originl_video, model.predict, segments)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], prime_frame_num, num_features=15, hide_rest=False)

    t = plt.imshow(mark_boundaries(prime_frame_img, mask))
    t.show()


if __name__=='__main__':
    # start by removing masked videos from previous runs 
    for f in glob.glob('../../data/LIMEset/*'):
        os.remove(f)
    sys.exit()
    parser = argparse.ArgumentParser(description = "Description")
    parser.add_argument("-m", "--model", help = "Video classification model", required = False, default = "")
    parser.add_argument("-v", "--video", help = "Video to explain", required = True, default = "")

    argument = parser.parse_args()
    status = False

    if argument.model:
        print("You have used '-m' or '--model' with argument: {0}".format(argument.model))
        model = keras.models.load_model(arguement.model)
        status = True
    else:
        model = keras.models.load_model('../../data/models/predict_model')

    if argument.video:
        print("You have used '-v' or '--video' with argument: {0}".format(argument.video))
        status = True

    if not status:
        print("Maybe you want to use -m or -v as arguments ?")

    sys.exit()
