# 🏀 Basketball Video Analysis
Analyze basketball footage with automated detection of players, ball, team assignment, and more. This repository integrates object tracking, zero-shot classification, and custom keypoint detection for a fully annotated basketball game experience.

Leveraging the convenience of Roboflow for dataset management and Ultralytics' YOLO models for both training and inference, this project provides a robust framework for basketball video analysis.

Training notebooks are included to help you customize and fine-tune models to suit your specific needs, ensuring a seamless and efficient workflow.

# ✨ Features
- Player and ball detection/tracking using pretrained models.
- Court keypoint detection for visualizing important zones.
- Team assignment with jersey color classification.
- Ball possession detection, pass detection, and interception detection.
- Easy stubbing to skip repeated computation for fast iteration.
- Various “drawers” to overlay detected elements onto frames.

# 🏰 Project Structure
main.py
– Orchestrates the entire pipeline: reading video frames, running detection/tracking, team assignment, drawing results, and saving the output video.

trackers/
– Houses PlayerTracker and BallTracker, which use detection models to generate bounding boxes and track objects across frames.

utils/
– Contains helper functions like bbox_utils.py for geometric calculations, stubs_utils.py for reading and saving intermediate results, and video_utils.py for reading/saving videos.

drawers/
– Contains classes that overlay bounding boxes, court lines, passes, etc., onto frames.

ball_aquisition/
– Logic for identifying which player is in possession of the ball.

pass_and_interception_detector/
– Identifies passing events and interceptions.

court_keypoint_detector/
– Detects lines and keypoints on the court using the specified model.

team_assigner/
– Uses zero-shot classification (Hugging Face or similar) to assign players to teams based on jersey color.

configs/
– Holds default paths for models, stubs, and output video.

# ⚙️ Installation
Setup your environment locally.

Python Environment
Create a virtual environment (e.g., venv/conda).
Install the required packages:

``` pip install -r requirements.txt ```

*** Notice: May need to update versions of packages in requirements

# 🎓 Training the Models

This repository relies on trained models for detecting basketballs, players, and court keypoints. You have two options to get these models:

1. Download the Pretrained Weights

ball_detector_model.pt
(https://drive.google.com/file/d/1KejdrcEnto2AKjdgdo1U1syr5gODp6EL/view?usp=sharing)

court_keypoint_detector.pt
(https://drive.google.com/file/d/1nGoG-pUkSg4bWAUIeQ8aN6n7O1fOkXU0/view?usp=sharing)

player_detector.pt
(https://drive.google.com/file/d/1fVBLZtPy9Yu6Tf186oS4siotkioHBLHy/view?usp=sharing)

Simply download these files and place them into the models/ folder in your project. This allows you to run the pipelines without manually retraining.

2. Train Your Own Models
The training scripts are provided in the training_notebooks/ folder. These Jupyter notebooks use Roboflow datasets and the Ultralytics YOLO frameworks to train various detection tasks:

basketball_ball_training.ipynb: Trains a basketball ball detector (using YOLOv5). Incorporates motion blur augmentations to improve ball detection accuracy on fast-moving game footage.
basketball_court_keypoint_training.ipynb: Uses YOLOv8 to detect keypoints on the court (e.g., lines, corners, key zones).
basketball_player_detection_training.ipynb: Trains a player detection model (using YOLO v11) to identify players in each frame.
You can easily run these notebooks in Google Colab or another environment with GPU access. After training, download the newly generated .pt files and place them in the models/ folder.

Once you have your models in place, you may proceed with the usage steps described above. If you want to retrain or fine-tune for your specific dataset, remember to adjust the paths in the notebooks and in main.py to point to the newly generated models.

