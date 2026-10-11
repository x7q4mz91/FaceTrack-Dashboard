# FaceTrack Dashboard

A local Windows webcam face tracker with a password-protected dashboard for subject snapshots and face-triggered footage.

## Requirements

- Windows 10 or 11
- Python 3.11 (64-bit)
- A webcam

## Setup

1. Download or clone this repository and open PowerShell in the project folder.
2. Create a virtual environment and install the dependency:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Set a private dashboard password of at least 12 characters. This setting applies to the current PowerShell window:

   ```powershell
   $env:FACE_TRACKER_PASSWORD = "choose-a-private-password"
   ```

4. Start the tracker:

   ```powershell
   .\.venv\Scripts\python.exe ".\python face_tracker.py"
   ```

   The tracker opens the dashboard at <http://127.0.0.1:8765/>. If you do not set `FACE_TRACKER_PASSWORD`, it generates a temporary password and prints it in the PowerShell window each time it starts.

To keep a password for future PowerShell windows, run `setx FACE_TRACKER_PASSWORD "choose-a-private-password"` once, then open a new PowerShell window. Keep the password private and use at least 12 characters.

Press **Q** in the tracker window to stop the camera and website.

## Dashboard and recordings

- **Subjects** shows saved face snapshots. Edit each profile to change its name, notes, or status.
- **Footage** shows WebM clips recorded while one or more faces are detected. Clips are split into one-minute files and finalized shortly after faces leave view.
- Recordings and subject data are kept in the local `data/` folder. Use **Remove** or **Remove all footage** in the dashboard to delete recordings.
- The website listens only on `127.0.0.1` (this computer); it is not published to the internet or made available to other computers on your network.
- Allow camera access when prompted. The camera is accessed by the local Python application.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_tracking test_local_site
```

## Privacy

Face snapshots and video can contain sensitive personal information. They remain in the local `data/` directory; do not share that directory or its contents. The repository ignores this folder, virtual environments, local environment files, and video files. Record people only with appropriate notice and permission, and check local laws before using face recognition or recording.

This project uses face detection and an experimental appearance matcher; it can miss or confuse people. Do not rely on it for security, identity verification, or decisions about people.
