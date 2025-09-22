from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer,
    TeamBallControlDrawer,
    PassInterceptionDrawer,
    CourtKeypointDrawer,
    TacticalViewDrawer,
    SpeedAndDistanceDrawer
)
from team_assigner import TeamAssigner
from ball_acquisition import BallAcquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from court_keypoint_detector import CourtKeypointDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator

def main():
    # Read Video
    video_frames = read_video("input_videos/video_2.mp4")

    # Initialize Tracker
    player_tracker = PlayerTracker("models/player_detector.pt")
    ball_tracker = BallTracker("models/ball_detector_model.pt")

    # Initialize Court Keypoint Detector
    court_keypoint_detector = CourtKeypointDetector("models/court_keypoint_detector.pt")

    # Run Tracks
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="stubs/player_track_stubs.pkl")
    
    ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                                 read_from_stub=True,
                                                 stub_path="stubs/ball_tracker_stubs.pkl")
    

    # Get Court Keypoints
    court_keypoints = court_keypoint_detector.get_court_keypoints(video_frames,
                                                                  read_from_stub=True,
                                                                  stub_path="stubs/court_key_points_detector.pkl"
                                                                  )
    
    
    # Remove wrong ball Detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    
    # Interpolate Ball Tracks
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames,
                                                                player_tracks,
                                                                read_from_stub=True,
                                                                stub_path="stubs/player_assignment_stub.pkl"
                                                                )
    
    # Ball Acquisition
    ball_acquisition_detector = BallAcquisitionDetector()
    ball_acquisition = ball_acquisition_detector.detect_ball_possession(player_tracks, ball_tracks)

    # Detect Passes and Interceptions
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_acquisition, player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_acquisition, player_assignment)

    # Tactical View
    tactical_view_converter = TacticalViewConverter(court_image_path="./images/basketball_court.png")
    court_keypoints = tactical_view_converter.validate_keypoints(court_keypoints)
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_keypoints, player_tracks)

    # Speed and distance Calculator
    speed_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters
    )
    player_distance_per_frame = speed_distance_calculator.calculate_distance(tactical_player_positions)
    player_speed_per_frame = speed_distance_calculator.calculate_speed(player_distance_per_frame)


    # Draw Output
    # Initialize Drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    pass_interception_drawer = PassInterceptionDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    speed_and_distance_drawer = SpeedAndDistanceDrawer()

    # Draw Object Tracks
    output_video_frames = player_tracks_drawer.draw(video_frames,
                                                    player_tracks,
                                                    player_assignment,
                                                    ball_acquisition)
    
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    # Draw Team Ball Control
    output_video_frames = team_ball_control_drawer.draw(output_video_frames,
                                                        player_assignment,
                                                        ball_acquisition)
    
    # Draw Passes and Interceptions
    output_video_frames = pass_interception_drawer.draw(output_video_frames,
                                                        passes,
                                                        interceptions)
    
    # Draw Court Keypoints
    output_video_frames = court_keypoint_drawer.draw(output_video_frames, court_keypoints)

    # Tactical View
    output_video_frames = tactical_view_drawer.draw(output_video_frames,
                                                    tactical_view_converter.court_image_path,
                                                    tactical_view_converter.width,
                                                    tactical_view_converter.height,
                                                    tactical_view_converter.key_points,
                                                    tactical_player_positions,
                                                    player_assignment,
                                                    ball_acquisition,
                                                    )
    
    # Speed and Distance Drawer
    output_video_frames = speed_and_distance_drawer.draw(
        output_video_frames,
        player_tracks,
        player_distance_per_frame,
        player_speed_per_frame
    )

    # Save Video
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()