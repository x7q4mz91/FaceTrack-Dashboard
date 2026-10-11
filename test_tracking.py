import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


TRACKER_PATH = Path(__file__).with_name("python face_tracker.py")
SPEC = importlib.util.spec_from_file_location("face_tracker", TRACKER_PATH)
tracker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tracker)


class TrackingTests(unittest.TestCase):
    def test_suspicious_status_is_red_and_alerts_only_on_transition(self):
        track = {}
        self.assertEqual(tracker.status_color("Suspicious"), (0, 0, 255))
        self.assertTrue(tracker.should_play_alert(track, "Suspicious"))
        self.assertFalse(tracker.should_play_alert(track, "Suspicious"))
        self.assertFalse(tracker.should_play_alert(track, "Recognized"))
        self.assertTrue(tracker.should_play_alert(track, "Suspicious"))

    def test_saved_profile_match_requires_consistent_votes(self):
        track = {}
        self.assertIsNone(tracker.update_identity_vote(track, 7))
        self.assertIsNone(tracker.update_identity_vote(track, 7))
        self.assertIsNone(tracker.update_identity_vote(track, None))
        self.assertIsNone(tracker.update_identity_vote(track, 7))
        self.assertIsNone(tracker.update_identity_vote(track, 7))
        self.assertEqual(tracker.update_identity_vote(track, 7), 7)
        self.assertEqual(tracker.update_identity_vote(track, None), 7)
        self.assertEqual(tracker.update_identity_vote(track, 9), 7)

    def test_recognized_track_cannot_be_saved_as_a_new_subject(self):
        track = {
            "hits": tracker.NEW_SUBJECT_CONFIRMATION_HITS,
            "best_snapshot": b"snapshot",
        }
        for _ in range(tracker.UNKNOWN_PROFILE_STREAK):
            tracker.update_identity_vote(track, None)
        self.assertTrue(tracker.should_register_new_subject(track))

        for _ in range(tracker.IDENTITY_VOTE_COUNT):
            tracker.update_identity_vote(track, 7)
        self.assertFalse(tracker.should_register_new_subject(track))
        self.assertEqual(tracker.update_identity_vote(track, 7), 7)

        for _ in range(tracker.UNKNOWN_PROFILE_STREAK + 5):
            tracker.update_identity_vote(track, None)
        self.assertFalse(tracker.should_register_new_subject(track))

    def test_tentative_profile_vote_does_not_permanently_block_new_subject(self):
        track = {
            "hits": tracker.NEW_SUBJECT_CONFIRMATION_HITS,
        }
        tracker.update_identity_vote(track, 7)
        for _ in range(tracker.UNKNOWN_PROFILE_STREAK):
            tracker.update_identity_vote(track, None)
        self.assertTrue(tracker.should_register_new_subject(track))

    def test_unconfirmed_track_waits_before_creating_profile(self):
        track = {"hits": tracker.CONFIRMATION_HITS}
        for _ in range(tracker.CONFIRMATION_HITS):
            tracker.update_identity_vote(track, None)
        self.assertFalse(tracker.should_register_new_subject(track))

        track["hits"] = tracker.NEW_SUBJECT_CONFIRMATION_HITS
        for _ in range(
            tracker.UNKNOWN_PROFILE_STREAK - tracker.CONFIRMATION_HITS
        ):
            tracker.update_identity_vote(track, None)
        self.assertTrue(tracker.should_register_new_subject(track))

    def test_profile_matcher_recognizes_a_saved_face_template(self):
        face = np.zeros((128, 128), dtype=np.uint8)
        cv2 = tracker.cv2
        cv2.rectangle(face, (24, 18), (103, 109), 170, -1)
        cv2.circle(face, (50, 53), 8, 20, -1)
        cv2.circle(face, (78, 53), 8, 20, -1)
        cv2.ellipse(face, (64, 83), (20, 8), 0, 0, 180, 25, 3)
        encoded, reference = cv2.imencode(".jpg", face)
        self.assertTrue(encoded)

        class SavedProfile:
            active_subjects = []

            def list_subjects(self):
                return self.active_subjects

            def read_image(self, subject_id):
                return reference.tobytes() if subject_id == 7 else None

            def read_reference_images(self, subject_id):
                return []

            def add_reference_image(self, subject_id, image_data):
                return False

        profile_store = SavedProfile()
        class FullImageDetector:
            def detectMultiScale(self, image, **kwargs):
                return [(0, 0, image.shape[1], image.shape[0])]

        matcher = tracker.PersistentFaceMatcher(profile_store, FullImageDetector())
        profile_store.active_subjects = [{"subject_id": 7}]
        self.assertTrue(matcher.add_reference(7, reference.tobytes()))
        self.assertEqual(len(matcher._reference_faces[7]), 1)
        self.assertEqual(matcher.match(face, (0, 0, 128, 128)), 7)
        profile_store.list_subjects = lambda: []
        self.assertIsNone(matcher.match(face, (0, 0, 128, 128)))

    def test_matcher_retrains_with_confirmed_reference_images(self):
        cv2 = tracker.cv2
        face = np.zeros((128, 128), dtype=np.uint8)
        cv2.rectangle(face, (24, 18), (103, 109), 170, -1)
        cv2.circle(face, (50, 53), 8, 20, -1)
        cv2.circle(face, (78, 53), 8, 20, -1)
        cv2.ellipse(face, (64, 83), (20, 8), 0, 0, 180, 25, 3)
        encoded, reference = cv2.imencode(".jpg", face)
        self.assertTrue(encoded)

        class SavedProfile:
            references = []

            def list_subjects(self):
                return [{"subject_id": 7}]

            def read_image(self, subject_id):
                return reference.tobytes() if subject_id == 7 else None

            def read_reference_images(self, subject_id):
                return list(self.references) if subject_id == 7 else []

            def add_reference_image(self, subject_id, image_data):
                self.references.append(image_data)
                return True

        class FullImageDetector:
            def detectMultiScale(self, image, **kwargs):
                return [(0, 0, image.shape[1], image.shape[0])]

        profile_store = SavedProfile()
        matcher = tracker.PersistentFaceMatcher(
            profile_store,
            FullImageDetector(),
        )
        variation = cv2.convertScaleAbs(face, alpha=0.9, beta=10)
        encoded_variation, variation_bytes = cv2.imencode(".jpg", variation)
        self.assertTrue(encoded_variation)
        self.assertTrue(matcher.add_reference(7, variation_bytes.tobytes()))
        self.assertEqual(len(matcher._reference_faces[7]), 2)
        self.assertEqual(matcher.match(variation, (0, 0, 128, 128)), 7)

        reopened_matcher = tracker.PersistentFaceMatcher(
            profile_store,
            FullImageDetector(),
        )
        self.assertEqual(len(reopened_matcher._reference_faces[7]), 2)
        self.assertEqual(reopened_matcher.match(variation, (0, 0, 128, 128)), 7)

    def test_reference_extraction_prefers_centered_subject_in_multi_face_crop(self):
        image = np.zeros((200, 200), dtype=np.uint8)
        image[10:90, 10:90] = 40
        image[75:125, 75:125] = 180

        class MultipleFaceDetector:
            def detectMultiScale(self, image, **kwargs):
                return [(10, 10, 80, 80), (75, 75, 50, 50)]

        matcher = tracker.PersistentFaceMatcher.__new__(
            tracker.PersistentFaceMatcher
        )
        matcher._detector = MultipleFaceDetector()
        extracted = matcher._extract_reference(image)
        expected = tracker.normalize_face(image, (75, 75, 50, 50))
        np.testing.assert_array_equal(extracted, expected)

    def test_multi_person_tracks_use_global_assignment(self):
        tracks = {
            1: {
                "box": (50, 50, 100, 100),
                "missed": 0,
                "hits": 10,
                "velocity": (0.0, 0.0),
            },
            2: {
                "box": (70, 50, 100, 100),
                "missed": 0,
                "hits": 10,
                "velocity": (0.0, 0.0),
            },
        }
        faces = [(59, 50, 100, 100), (50, 65, 100, 100)]

        ids, updated_tracks, next_id = tracker.match_faces(faces, tracks, 3)

        self.assertEqual(ids, [2, 1])
        self.assertEqual(next_id, 3)
        self.assertEqual(updated_tracks[1]["hits"], 11)
        self.assertEqual(updated_tracks[2]["hits"], 11)

    def test_footage_recorder_records_only_face_presence_and_finalizes_clip(self):
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            partial = Path(temp_dir) / "footage-active-partial.webm"
            final = Path(temp_dir) / "footage-saved.webm"

            class Store:
                def new_footage_paths(self):
                    return partial, final

            class FakeWriter:
                def __init__(self, path):
                    self.path = Path(path)
                    self.frame_count = 0

                def isOpened(self):
                    return True

                def write(self, frame):
                    self.frame_count += 1

                def release(self):
                    self.path.write_bytes(b"recorded clip")

            writers = []

            def create_writer(path, codec, fps, size):
                writer = FakeWriter(path)
                writers.append(writer)
                return writer

            recorder = tracker.FootageRecorder(Store(), fps=2)
            with patch.object(tracker.cv2, "VideoWriter", side_effect=create_writer):
                recorder.process(frame, has_faces=False)
                self.assertEqual(writers, [])
                recorder.process(frame, has_faces=True)
                recorder.process(frame, has_faces=False)
                recorder.process(frame, has_faces=False)
                recorder.process(frame, has_faces=False)
                self.assertFalse(final.exists())
                recorder.process(frame, has_faces=False)
                self.assertEqual(final.read_bytes(), b"recorded clip")
                self.assertEqual(writers[0].frame_count, 4)
                recorder.close()

    def test_track_id_survives_motion_and_temporary_missed_detections(self):
        ids, tracks, next_id = tracker.match_faces([(100, 100, 100, 100)], {}, 1)
        subject_id = ids[0]

        for step in range(1, 8):
            ids, tracks, next_id = tracker.match_faces(
                [(100 + step * 5, 100, 100, 100)], tracks, next_id
            )
            self.assertEqual(ids, [subject_id])

        for _ in range(30):
            ids, tracks, next_id = tracker.match_faces([], tracks, next_id)

        ids, tracks, _ = tracker.match_faces([(150, 100, 100, 100)], tracks, next_id)
        self.assertEqual(ids, [subject_id])

    def test_low_quality_dark_snapshot_is_rejected(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertIsNone(tracker.make_snapshot_candidate(frame, (200, 120, 120, 140)))


if __name__ == "__main__":
    unittest.main()
