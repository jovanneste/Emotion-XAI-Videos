from lime_video import *
from mask import *
import sys
import argparse


def explain_model_prediction(video_path, model, num_features, num_segments, verbose):
    label = 0 #label to explain (0-exciting, 1-funny)
    n = 15
    prime_frame, lower_frame, upper_frame, frameSize, fps = maskFrames(video_path, n, model, verbose, label)
    if verbose:
        print("Frames")
        print(prime_frame, lower_frame, upper_frame)
        print("Creating masks...")

    createMaskedVideos(prime_frame, lower_frame, upper_frame-1, fps, frameSize, 20, num_segments, verbose)

    file = open('segments_and_prime_frame', 'rb')
    segments_and_prime_frame = pickle.load(file)
    segments = segments_and_prime_frame[0]
    prime_frame_num = segments_and_prime_frame[1]
    file.close()

    originl_video = '../../data/LIMEset/0.mp4'
    prime_frame_img = cv2.imread('../../data/frames/frame'+str(prime_frame_num)+'.jpg')[:,:,::-1]

    explainer = LimeVideoExplainer(label)
    explanation = explainer.explain_instances(originl_video, model.predict, segments)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0][label], prime_frame_num, num_features)

    if verbose:
        print("Creating LIME explainer...")
        plt.imshow(mask)
        plt.show()
        plt.imshow(prime_frame_img)
        plt.show()
        plt.imshow(mark_boundaries(prime_frame_img, mask))
        plt.show()

    plt.imsave('output.jpg', mark_boundaries(prime_frame_img, mask))

    # for f in glob.glob():
    #     if f.endswith('mp4'):
    #         print(f)
    #         os.remove(f)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description = "Usage")
    parser.add_argument("-m", "--model", help = "Video classification model", required = True, default = "")
    parser.add_argument("-v", "--video", help = "Video to explain", required = True, default = "")
    parser.add_argument("-s", "--segments", help = "Number of segments to split key frame into", required = False, default = 50)
    parser.add_argument("-f", "--features", help = "Number of features to display", required = False, default = 3)
    parser.add_argument("-p", "--print", help = "Verbose output", required = False, default = False)

    argument = parser.parse_args()
    status = False

    if argument.model:
        print("You have used '-m' or '--model' with argument: {0}".format(argument.model))
        status = True

    if argument.video:
        print("You have used '-v' or '--video' with argument: {0}".format(argument.video))
        status = True

    if argument.segments:
        print("You have used '-s' or '--segments' with argument: {0}".format(argument.segments))
        status = True

    if argument.features:
        print("You have used '-f' or '--features' with argument: {0}".format(argument.features))
        status = True

    if argument.print:
        print("You have used '-p' or '--print' with argument: {0}".format(argument.print))
        status = True

    if not status:
        print("Model and video are required arguements")
        sys.exit()

    # start by removing masked videos and pickle file from previous run
    for f in glob.glob('../../data/LIMEset/*'):
        os.remove(f)
    try:
        os.remove('segments_and_prime_frame')
    except:
        print('Removed')

    model = keras.models.load_model(argument.model)
    data = load_sample(argument.video)
    print("Video result", predict(data, model))


    # parameters: video, model, features to show, pixel segments, verbose
    explain_model_prediction(argument.video, model, int(argument.features), int(argument.segments), argument.print)
