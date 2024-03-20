import logging
import os
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.append(r"C:\Users\HARSHITA KAMANI\Desktop\l2n\models\slide_classifier")
import inference  # pylint: disable=wrong-import-position
from custom_nnmodules import *  # noqa: F403,F401
from helpers import make_dir_if_not_exist

logger = logging.getLogger(__name__)

def classify_frames(
    frames_dir=r"C:\Users\HARSHITA KAMANI\Desktop\l2n\frames", do_copy=True, incorrect_threshold=0.60, model_path=r"C:\Users\HARSHITA KAMANI\Desktop\l2n\model_best.ckpt"
):
    model = inference.load_model(model_path)

    certainties = []
    frames_sorted_dir = Path(frames_dir).parent / "frames_sorted_1"
    logger.debug("Received inputs:\nframes_dir=" + str(frames_dir))

    frames = os.listdir(frames_dir)
    num_frames = len(frames)
    num_incorrect = 0
    percent_wrong = 0

    logger.info("Ready to classify " + str(num_frames) + " frames")

    frames_tqdm = tqdm(enumerate(frames), total=len(frames), desc="Classifying Frames")
    for idx, frame in frames_tqdm:
        current_frame_path = os.path.join(frames_dir, frame)
        best_guess, best_guess_idx, probs, _ = inference.get_prediction(
            model, Image.open(current_frame_path), extract_features=False
        )
        prob_max_correct = list(probs.values())[best_guess_idx]
        certainties.append(prob_max_correct)
        logger.debug("Prediction is " + best_guess)
        logger.debug("Probabilities are " + str(probs))
        if prob_max_correct < incorrect_threshold:
            num_incorrect = num_incorrect + 1
            percent_wrong = (num_incorrect / num_frames) * 100

            frames_tqdm.set_postfix(
                {"num_incorrect": num_incorrect, "percent_wrong": int(percent_wrong)}
            )

        if do_copy:
            classified_image_dir = frames_sorted_dir / best_guess
            make_dir_if_not_exist(classified_image_dir)
            shutil.copy(str(current_frame_path), str(classified_image_dir))

    logger.info("Percent frames classified incorrectly: " + str(percent_wrong))
    logger.debug("Returning frames_sorted_dir=" + str(frames_sorted_dir))

    return frames_sorted_dir, certainties, percent_wrong

if __name__ == "__main__":
    frames_sorted_dir, certainties, percent_wrong = classify_frames()
    print(f"Frames sorted directory: {frames_sorted_dir}")
    #print(f"Certainties: {certainties}")
    print(f"Percent frames classified incorrectly: {percent_wrong}%")
