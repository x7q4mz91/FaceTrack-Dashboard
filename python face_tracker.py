import cv2
import numpy as np
import os
import winsound

from local_site import SubjectStore, start_site, stop_site


TRACK_MEMORY_FRAMES = 180
CONFIRMATION_HITS = 12
MIN_FACE_SIZE = 80
MIN_SNAPSHOT_SHARPNESS = 35.0
PROFILE_MATCH_THRESHOLD = 70.0
PROFILE_MATCH_MARGIN = 8.0
IDENTITY_VOTE_WINDOW = 8
IDENTITY_VOTE_COUNT = 5
NEW_SUBJECT_CONFIRMATION_HITS = 30
UNKNOWN_PROFILE_STREAK = 20
MAX_REFERENCE_IMAGES = 8
FOOTAGE_SEGMENT_SECONDS = 60
FOOTAGE_NO_FACE_GRACE_SECONDS = 1.5


def status_color(status):
    return (0, 0, 255) if status == "Suspicious" else (255, 220, 40)


def should_play_alert(track, status):
    previous_status = track.get("status")
    track["status"] = status
    return status == "Suspicious" and previous_status != "Suspicious"


def normalize_face(gray, box):
    x, y, width, height = box
    face = gray[y:y + height, x:x + width]
    if face.size == 0:
        return None
    return cv2.resize(cv2.equalizeHist(face), (128, 128), interpolation=cv2.INTER_AREA)


class PersistentFaceMatcher:
    def __init__(self, store, face_detector):
        if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            raise RuntimeError("Install the opencv-contrib-python package to enable saved face matching.")

        self._store = store
        self._detector = face_detector
        self._recognizers = {}
        self._reference_faces = {}
        for subject in store.list_subjects():
            subject_id = int(subject["subject_id"])
            image_bytes = [store.read_image(subject_id)]
            image_bytes.extend(store.read_reference_images(subject_id))
            for image_data in image_bytes:
                if image_data is None:
                    continue
                image = cv2.imdecode(
                    np.frombuffer(image_data, dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE,
                )
                if image is None:
                    continue
                face = self._extract_reference(image)
                if face is not None:
                    self._add_reference(subject_id, face)

    def _extract_reference(self, gray_image):
        detections = self._detector.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(MIN_FACE_SIZE // 2, MIN_FACE_SIZE // 2),
        )
        if len(detections) == 0:
            return None
        image_center = (gray_image.shape[1] / 2, gray_image.shape[0] / 2)
        face_box = min(
            detections,
            key=lambda rect: (
                ((rect[0] + rect[2] / 2 - image_center[0]) ** 2
                 + (rect[1] + rect[3] / 2 - image_center[1]) ** 2)
                / max(1, rect[2] * rect[3]),
                -int(rect[2]) * int(rect[3]),
            ),
        )
        return normalize_face(gray_image, tuple(int(value) for value in face_box))

    def _add_reference(self, subject_id, face):
        references = self._reference_faces.setdefault(subject_id, [])
        if len(references) >= MAX_REFERENCE_IMAGES + 1:
            return
        references.append(face)
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=8,
            grid_x=8,
            grid_y=8,
            threshold=PROFILE_MATCH_THRESHOLD,
        )
        recognizer.train(
            references,
            np.full(len(references), subject_id, dtype=np.int32),
        )
        self._recognizers[subject_id] = recognizer

    def add_reference(self, subject_id, snapshot_bytes):
        image = cv2.imdecode(
            np.frombuffer(snapshot_bytes, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None:
            raise RuntimeError(f"Could not decode the saved snapshot for Subject_{subject_id:03d}.")
        face = self._extract_reference(image)
        references = self._reference_faces.get(subject_id, [])
        if (
            face is not None
            and not references
            and self._store.read_image(subject_id) == snapshot_bytes
        ):
            self._add_reference(subject_id, face)
            return True
        if face is not None and len(references) <= MAX_REFERENCE_IMAGES:
            if self._store.add_reference_image(subject_id, snapshot_bytes):
                self._add_reference(subject_id, face)
                return True
        return False

    def match(self, gray_frame, box):
        face = normalize_face(gray_frame, box)
        if face is None:
            return None

        active_subject_ids = {
            int(subject["subject_id"]) for subject in self._store.list_subjects()
        }
        candidates = []
        for subject_id, recognizer in self._recognizers.items():
            if subject_id not in active_subject_ids:
                continue
            _, confidence = recognizer.predict(face)
            candidates.append((float(confidence), subject_id))
        if not candidates:
            return None

        candidates.sort()
        best_confidence, best_subject_id = candidates[0]
        second_confidence = candidates[1][0] if len(candidates) > 1 else float("inf")
        if (
            best_confidence <= PROFILE_MATCH_THRESHOLD
            and second_confidence - best_confidence >= PROFILE_MATCH_MARGIN
        ):
            return best_subject_id
        return None


def update_identity_vote(track, candidate_subject_id):
    votes = track.setdefault("identity_votes", [])
    votes.append(candidate_subject_id)
    del votes[:-IDENTITY_VOTE_WINDOW]
    if candidate_subject_id is None:
        track["unmatched_streak"] = track.get("unmatched_streak", 0) + 1
        return track.get("persistent_subject_id")

    track["unmatched_streak"] = 0
    if votes.count(candidate_subject_id) >= IDENTITY_VOTE_COUNT:
        track["persistent_subject_id"] = candidate_subject_id
    return track.get("persistent_subject_id")


def should_register_new_subject(track):
    return (
        track.get("persistent_subject_id") is None
        and track.get("hits", 0) >= NEW_SUBJECT_CONFIRMATION_HITS
        and track.get("unmatched_streak", 0) >= UNKNOWN_PROFILE_STREAK
    )


def _minimum_cost_assignment(cost_matrix):
    row_count = len(cost_matrix)
    if row_count == 0:
        return []
    column_count = len(cost_matrix[0])
    if column_count < row_count:
        raise ValueError("Assignment matrix must have at least as many columns as rows.")

    row_potentials = [0.0] * (row_count + 1)
    column_potentials = [0.0] * (column_count + 1)
    matched_rows = [0] * (column_count + 1)
    previous_columns = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        matched_rows[0] = row
        current_column = 0
        minimum_costs = [float("inf")] * (column_count + 1)
        visited = [False] * (column_count + 1)
        while True:
            visited[current_column] = True
            current_row = matched_rows[current_column]
            delta = float("inf")
            next_column = 0
            for column in range(1, column_count + 1):
                if visited[column]:
                    continue
                reduced_cost = (
                    cost_matrix[current_row - 1][column - 1]
                    - row_potentials[current_row]
                    - column_potentials[column]
                )
                if reduced_cost < minimum_costs[column]:
                    minimum_costs[column] = reduced_cost
                    previous_columns[column] = current_column
                if minimum_costs[column] < delta:
                    delta = minimum_costs[column]
                    next_column = column

            for column in range(column_count + 1):
                if visited[column]:
                    row_potentials[matched_rows[column]] += delta
                    column_potentials[column] -= delta
                else:
                    minimum_costs[column] -= delta
            current_column = next_column
            if matched_rows[current_column] == 0:
                break

        while True:
            previous_column = previous_columns[current_column]
            matched_rows[current_column] = matched_rows[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    assignments = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_rows[column] != 0:
            assignments[matched_rows[column] - 1] = column - 1
    return assignments


def match_faces(faces, tracks, next_id):
    track_ids = sorted(tracks)
    unmatched_cost = 1.1251
    invalid_cost = 1_000_000.0
    costs = []

    for track_id in track_ids:
        track = tracks[track_id]
        old_x, old_y, old_w, old_h = track["box"]
        velocity_x, velocity_y = track.get("velocity", (0.0, 0.0))
        missed = track.get("missed", 0)
        old_center = (
            old_x + old_w / 2 + velocity_x * min(missed + 1, 10),
            old_y + old_h / 2 + velocity_y * min(missed + 1, 10),
        )

        row_costs = []
        for index, (x, y, width, height) in enumerate(faces):
            center = (x + width / 2, y + height / 2)
            distance = ((old_center[0] - center[0]) ** 2 + (old_center[1] - center[1]) ** 2) ** 0.5
            max_distance = max(old_w, old_h, width, height) * 1.8
            size_ratio = max(old_w * old_h, width * height) / max(1, min(old_w * old_h, width * height))
            cost = distance / max_distance + size_ratio * 0.05
            row_costs.append(
                cost if distance <= max_distance and size_ratio <= 2.5 else invalid_cost
            )
        costs.append(row_costs + [unmatched_cost] * len(track_ids))

    assignments = [None] * len(faces)
    used_tracks = set()
    for track_row, column in enumerate(_minimum_cost_assignment(costs)):
        if column < len(faces) and costs[track_row][column] < unmatched_cost:
            track_id = track_ids[track_row]
            assignments[column] = track_id
            used_tracks.add(track_id)

    updated_tracks = {}
    for index, face in enumerate(faces):
        track_id = assignments[index]

        if track_id is None:
            track_id = next_id
            next_id += 1
            updated_tracks[track_id] = {
                "box": face,
                "missed": 0,
                "hits": 1,
                "velocity": (0.0, 0.0),
                "best_snapshot": None,
                "best_score": 0.0,
                "identity_votes": [],
                "unmatched_streak": 0,
            }
        else:
            previous = tracks[track_id]
            old_x, old_y, old_w, old_h = previous["box"]
            x, y, width, height = face
            old_center = (old_x + old_w / 2, old_y + old_h / 2)
            center = (x + width / 2, y + height / 2)
            previous_velocity = previous.get("velocity", (0.0, 0.0))
            velocity = (
                previous_velocity[0] * 0.5 + (center[0] - old_center[0]) * 0.5,
                previous_velocity[1] * 0.5 + (center[1] - old_center[1]) * 0.5,
            )
            updated_tracks[track_id] = {
                **previous,
                "box": face,
                "missed": 0,
                "hits": previous.get("hits", 0) + 1,
                "velocity": velocity,
            }

        assignments[index] = track_id

    for track_id, track in tracks.items():
        if track_id not in used_tracks:
            track["missed"] += 1
            if track["missed"] <= TRACK_MEMORY_FRAMES:
                updated_tracks[track_id] = track

    return assignments, updated_tracks, next_id


def make_snapshot_candidate(frame, box):
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = box
    if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
        return None

    padding_x = int(width * 0.2)
    padding_top = int(height * 0.35)
    padding_bottom = int(height * 0.15)
    left = max(0, x - padding_x)
    top = max(0, y - padding_top)
    right = min(frame_width, x + width + padding_x)
    bottom = min(frame_height, y + height + padding_bottom)
    crop = frame[top:bottom, left:right]
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray_crop.mean())
    sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    if not 45 <= brightness <= 215 or sharpness < MIN_SNAPSHOT_SHARPNESS:
        return None

    encoded, image_data = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not encoded:
        return None
    return sharpness, image_data.tobytes()


def draw_tracking_overlay(
    frame,
    box,
    subject_id,
    frame_number,
    status="Unclassified",
    display_name=None,
):
    x, y, width, height = box
    color = status_color(status)
    corner = max(12, min(width, height) // 5)
    thickness = 2

    cv2.line(frame, (x, y), (x + corner, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + corner), color, thickness)
    cv2.line(frame, (x + width, y), (x + width - corner, y), color, thickness)
    cv2.line(frame, (x + width, y), (x + width, y + corner), color, thickness)
    cv2.line(frame, (x, y + height), (x + corner, y + height), color, thickness)
    cv2.line(frame, (x, y + height), (x, y + height - corner), color, thickness)
    cv2.line(frame, (x + width, y + height), (x + width - corner, y + height), color, thickness)
    cv2.line(frame, (x + width, y + height), (x + width, y + height - corner), color, thickness)

    scan_y = y + 4 + int((frame_number % 30) / 30 * max(1, height - 8))
    cv2.line(frame, (x + 4, scan_y), (x + width - 4, scan_y), color, 1)

    label = display_name or f"Subject_{subject_id:03d}"
    label_y = max(0, y - 30)
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    cv2.rectangle(
        frame,
        (x, label_y),
        (x + label_width + 12, label_y + label_height + baseline + 8),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x + 6, label_y + label_height + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 30, 35),
        2,
    )
    if status == "Suspicious":
        cv2.putText(
            frame,
            "ALERT",
            (x, min(frame.shape[0] - 8, y + height + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )


class FootageRecorder:
    def __init__(self, store, fps):
        self._store = store
        self._fps = max(float(fps), 1.0)
        self._segment_frame_limit = max(1, int(self._fps * FOOTAGE_SEGMENT_SECONDS))
        self._no_face_frame_limit = max(1, int(self._fps * FOOTAGE_NO_FACE_GRACE_SECONDS))
        self._writer = None
        self._partial_path = None
        self._final_path = None
        self._segment_frames = 0
        self._no_face_frames = 0

    def process(self, frame, has_faces):
        if has_faces:
            self._no_face_frames = 0
            if self._writer is None or self._segment_frames >= self._segment_frame_limit:
                self._finish_segment()
                self._start_segment(frame)
            self._writer.write(frame)
            self._segment_frames += 1
            return

        if self._writer is None:
            return
        self._no_face_frames += 1
        if (
            self._no_face_frames > self._no_face_frame_limit
            or self._segment_frames >= self._segment_frame_limit
        ):
            self._finish_segment()
            return
        self._writer.write(frame)
        self._segment_frames += 1

    def _start_segment(self, frame):
        self._partial_path, self._final_path = self._store.new_footage_paths()
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(self._partial_path),
            cv2.VideoWriter_fourcc(*"VP80"),
            self._fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            self._partial_path.unlink(missing_ok=True)
            self._partial_path = None
            self._final_path = None
            raise RuntimeError("Could not start WebM footage recording with the VP8 codec.")
        self._writer = writer
        self._segment_frames = 0

    def _finish_segment(self):
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        if self._partial_path is None or self._final_path is None:
            raise RuntimeError("Footage recorder lost its active clip path.")
        if not self._partial_path.is_file() or self._partial_path.stat().st_size == 0:
            self._partial_path.unlink(missing_ok=True)
            raise RuntimeError("Footage recording produced an empty video clip.")
        try:
            os.replace(self._partial_path, self._final_path)
        except OSError as error:
            raise RuntimeError(
                f"Could not save footage clip to {self._final_path}."
            ) from error
        self._partial_path = None
        self._final_path = None
        self._segment_frames = 0
        self._no_face_frames = 0

    def close(self):
        self._finish_segment()


def main():
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_detector.empty():
        raise RuntimeError("Could not load OpenCV's frontal-face detector.")

    store = SubjectStore()
    profile_matcher = PersistentFaceMatcher(store, face_detector)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not access camera 0. Check camera permissions or connection.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 20)

    camera_fps = camera.get(cv2.CAP_PROP_FPS)
    recorder = FootageRecorder(store, camera_fps if camera_fps > 0 else 20)
    site_server = None
    tracks = {}
    next_id = store.next_subject_id()
    frame_number = 0

    try:
        site_server = start_site(store)
        print("Subject archive: http://127.0.0.1:8765/")

        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the camera.")
                break

            frame_number += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
            )

            boxes = [tuple(int(value) for value in face) for face in faces]
            subject_ids, tracks, next_id = match_faces(boxes, tracks, next_id)

            for box, subject_id in zip(boxes, subject_ids):
                track = tracks[subject_id]
                candidate_subject_id = profile_matcher.match(gray, box)
                persistent_subject_id = update_identity_vote(track, candidate_subject_id)
                display_subject_id = persistent_subject_id or subject_id

                if (
                    not store.has_subject(display_subject_id)
                    and track["hits"] >= CONFIRMATION_HITS
                ):
                    candidate = make_snapshot_candidate(frame, box)
                    if candidate is not None:
                        score, image_data = candidate
                        if score > track["best_score"]:
                            track["best_score"] = score
                            track["best_snapshot"] = image_data
                        if (
                            should_register_new_subject(track)
                            and track["best_snapshot"] is not None
                        ):
                            if not store.add_subject(display_subject_id, track["best_snapshot"]):
                                raise RuntimeError(
                                    f"Could not save Subject_{display_subject_id:03d}; "
                                    "that ID is already in use."
                                )
                            profile_matcher.add_reference(display_subject_id, track["best_snapshot"])
                            track["persistent_subject_id"] = display_subject_id
                elif (
                    persistent_subject_id is not None
                    and candidate_subject_id == persistent_subject_id
                    and track.get("reference_saved_for") != persistent_subject_id
                ):
                    candidate = make_snapshot_candidate(frame, box)
                    if candidate is not None:
                        _, image_data = candidate
                        if profile_matcher.add_reference(persistent_subject_id, image_data):
                            track["reference_saved_for"] = persistent_subject_id
                display_subject_id = track.get("persistent_subject_id") or subject_id
                status = store.subject_status(display_subject_id)
                if should_play_alert(tracks[subject_id], status):
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                display_name = (
                    store.subject_name(display_subject_id)
                    if store.has_subject(display_subject_id)
                    else "Identifying..."
                )
                draw_tracking_overlay(
                    frame,
                    box,
                    display_subject_id,
                    frame_number,
                    status,
                    display_name,
                )

            cv2.putText(
                frame,
                "FACE SCAN ACTIVE  |  Q TO QUIT",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 220, 40),
                2,
            )
            recorder.process(frame, bool(boxes))
            cv2.imshow("Face Tracker", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        try:
            recorder.close()
        finally:
            camera.release()
            cv2.destroyAllWindows()
            if site_server is not None:
                stop_site(site_server)


if __name__ == "__main__":
    main()
