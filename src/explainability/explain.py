from lime_video import *
from mask import *
import sys
import argparse
import cv2
import scipy
import random

def explain_model_prediction(video_path, model, num_features, num_segments, verbose, result):
    label = 0 #label to explain (0-exciting, 1-funny)
    n = 10
    fake_data_insts = 20
    prime_frame, lower_frame, upper_frame, frameSize, fps = maskFrames(video_path, n, model, verbose, label)
    if verbose:
        print("Frames")
        print(prime_frame, lower_frame, upper_frame)
        print("Creating masks...")

    createMaskedVideos(prime_frame, lower_frame, upper_frame-1, fps, frameSize, fake_data_insts, num_segments, verbose)
    num_frames = upper_frame-lower_frame

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

    plt.imsave('exlpanation.jpg', mark_boundaries(prime_frame_img, mask))


    masked_out = cv2.VideoWriter('masked.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), fps, frameSize)
    normal_out = cv2.VideoWriter('normal.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), fps, frameSize)
    mask_3d = np.dstack([mask]*3)
    mask_lime = np.where((mask_3d==0)|(mask_3d==1), mask_3d^1, mask_3d)

    frames = getSortedFrames()
    for frame in frames:
        if lower_frame <= prime_frame <= upper_frame-1:
            img = cv2.imread('../../data/frames/frame' +str(frame)+ '.jpg')
            normal_out.write(img)
            i = (img * mask_lime).clip(0, 255).astype(np.uint8)
            masked_out.write(i)

    normal_out.release()
    masked_out.release()


    masked_data = load_sample('masked.mp4')
    normal_data = load_sample('normal.mp4')

    masked_result = predict(masked_data, model)
    normal_result = predict(normal_data, model)


    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(normal_result, masked_result)


    print()
    print("Model prediction:", result)
    print("Explanation for label " + str(label) + " complete - saved as explanation.jpg")
    print("Stability", r_value**2)
    print("Exciting fidelity: " + str((normal_result[0]-masked_result[0])))
    print("Funny fidelity: " + str((normal_result[1]-masked_result[1])))
    # print(scipy.stats.pearsonr(video_result, shap_result))

    #---------------------------------------------------------------------------

    # print("Calculating random fidelity")

    feature_ids = list(np.unique(segments))
    random_features = random.sample(feature_ids, 3)
    segments_3d = np.dstack([segments]*3)

    mask_random = np.where((segments_3d==random_features[0])|(segments_3d==random_features[1]|(segments_3d==random_features[2])), 1, segments_3d)
    mask_random = np.where((mask_random!=1), 0, mask_random)

    random_out = cv2.VideoWriter('random.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), fps, frameSize)

    frames = getSortedFrames()
    for frame in frames:
        if lower_frame <= prime_frame <= upper_frame-1:
            img = cv2.imread('../../data/frames/frame' +str(frame)+ '.jpg')
            i = (img * mask_random).clip(0, 255).astype(np.uint8)
            random_out.write(i)


    random_out.release()


    random_data = load_sample('masked.mp4')

    random_result = predict(random_data, model)

    # print("Stability")
    # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(normal_result, random_result)
    # print(r_value**2)
    #
    # print("Exciting fidelity: " + str(normal_result[0]-random_result[0]))
    # print("Funny fidelity: " + str(normal_result[1]-random_result[1]))



if __name__=='__main__':
    parser = argparse.ArgumentParser(description = "Usage")
    parser.add_argument("-m", "--model", help = "Video classification model", required = True, default = "")
    parser.add_argument("-v", "--video", help = "Video to explain", required = True, default = "")
    parser.add_argument("-s", "--segments", help = "Number of segments to split key frame into", required = False, default = 50)
    parser.add_argument("-f", "--features", help = "Number of features to display", required = False, default = 2)
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
    result = predict(data, model)


    # parameters: video, model, features to show, pixel segments, verbose
    explain_model_prediction(argument.video, model, int(argument.features), int(argument.segments), argument.print, result)
