import datetime
import html
import http.cookies
import http.server
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser


HOST = "127.0.0.1"
PORT = 8765
MAX_FORM_BYTES = 8192
SUBJECT_STATUSES = ("Unclassified", "Recognized", "Suspicious")
MAX_REFERENCE_IMAGES = 8
FOOTAGE_FILE_PATTERN = re.compile(r"footage-\d{8}-\d{6}-[0-9a-f]{8}\.webm")
ARCHIVE_SCRIPT = """async function checkForNewSubjects() {
  if (/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) return;
  const response = await fetch("/api/count", { cache: "no-store" });
  if (!response.ok) return;
  const current = await response.json();
  const displayed = Number(document.getElementById("archive").dataset.count);
  if (current.count !== displayed) window.location.reload();
}
document.addEventListener("submit", (event) => {
  const message = event.target.dataset.confirm;
  if (message && !window.confirm(message)) event.preventDefault();
});
window.setInterval(() => checkForNewSubjects().catch((error) => {
  console.error("Could not refresh the subject archive:", error);
}), 2000);
"""
FOOTAGE_SCRIPT = """async function checkForNewFootage() {
  const response = await fetch("/api/footage-count", { cache: "no-store" });
  if (!response.ok) return;
  const current = await response.json();
  const displayed = Number(document.getElementById("footage").dataset.count);
  if (current.count !== displayed) window.location.reload();
}
document.addEventListener("submit", (event) => {
  const message = event.target.dataset.confirm;
  if (message && !window.confirm(message)) event.preventDefault();
});
window.setInterval(() => checkForNewFootage().catch((error) => {
  console.error("Could not refresh saved footage:", error);
}), 2000);
"""


class SubjectStore:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "data"
        self.images_dir = self.data_dir / "subjects"
        self.footage_dir = self.data_dir / "footage"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.footage_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self.data_dir / "subjects.json"
        self._next_id_path = self.data_dir / "next_subject_id.txt"
        self._lock = threading.RLock()

        if self._file_path.exists():
            try:
                subjects = json.loads(self._file_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(f"Could not load subject profiles from {self._file_path}.") from error
            if not isinstance(subjects, dict):
                raise RuntimeError(f"Subject profile data in {self._file_path} is invalid.")
            self._subjects = subjects
        else:
            self._subjects = {}
        highest_id = max((int(key) for key in self._subjects), default=0) + 1
        try:
            saved_next_id = int(self._next_id_path.read_text(encoding="ascii"))
        except FileNotFoundError:
            saved_next_id = highest_id
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Could not load the next subject ID from {self._next_id_path}.") from error
        self._next_id = max(highest_id, saved_next_id)
        if not self._next_id_path.exists():
            self._write_next_id(self._next_id)
    def next_subject_id(self):
        with self._lock:
            return self._next_id

    def has_subject(self, subject_id):
        with self._lock:
            return str(subject_id) in self._subjects

    def list_subjects(self):
        with self._lock:
            return [
                dict(self._subjects[key])
                for key in sorted(self._subjects, key=int)
                if not self._subjects[key].get("deleted")
            ]

    def add_subject(self, subject_id, jpeg_data):
        key = str(subject_id)
        if not key.isdecimal() or int(key) < 1:
            raise ValueError("Subject IDs must be positive integers.")
        if not jpeg_data:
            raise ValueError("Cannot save an empty subject image.")

        with self._lock:
            if key in self._subjects:
                return False

            image_name = f"subject_{int(key):03d}.jpg"
            image_path = self.images_dir / image_name
            self._write_image(image_path, jpeg_data)
            record = {
                "subject_id": int(key),
                "name": f"Subject_{int(key):03d}",
                "notes": "",
                "first_seen": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "image": image_name,
            }
            new_subjects = dict(self._subjects)
            new_subjects[key] = record
            self._write_json(new_subjects)
            next_id = max(self._next_id, int(key) + 1)
            if next_id != self._next_id:
                self._write_next_id(next_id)
                self._next_id = next_id
            self._subjects = new_subjects
            return True

    def add_reference_image(self, subject_id, jpeg_data):
        if not jpeg_data:
            raise ValueError("Cannot save an empty reference image.")

        key = str(subject_id)
        with self._lock:
            subject = self._subjects.get(key)
            if subject is None or subject.get("deleted"):
                return False

            references = list(subject.get("references", []))
            if len(references) >= MAX_REFERENCE_IMAGES:
                return False
            if (
                self.read_image(subject_id) == jpeg_data
                or jpeg_data in self.read_reference_images(subject_id)
            ):
                return False

            image_name = f"subject_{int(key):03d}_ref_{len(references) + 1:02d}.jpg"
            image_path = self.images_dir / image_name
            self._write_image(image_path, jpeg_data)
            new_subjects = dict(self._subjects)
            new_subjects[key] = {
                **subject,
                "references": [*references, image_name],
            }
            try:
                self._write_json(new_subjects)
            except OSError:
                image_path.unlink(missing_ok=True)
                raise
            self._subjects = new_subjects
            return True

    def _reference_path(self, subject_id, filename):
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.startswith(f"subject_{int(subject_id):03d}_ref_")
            or not filename.endswith(".jpg")
        ):
            raise RuntimeError(f"Invalid face reference filename for Subject_{int(subject_id):03d}.")
        return self.images_dir / filename

    def read_reference_images(self, subject_id):
        with self._lock:
            subject = self._subjects.get(str(subject_id))
            if subject is None or subject.get("deleted"):
                return []

            images = []
            for filename in subject.get("references", []):
                try:
                    images.append(self._reference_path(subject_id, filename).read_bytes())
                except OSError as error:
                    raise RuntimeError(
                        f"Could not read a face reference for Subject_{int(subject_id):03d}."
                    ) from error
            return images

    def update_profile(self, subject_id, name, notes, status):
        if status not in SUBJECT_STATUSES:
            raise ValueError("Invalid subject status.")
        key = str(subject_id)
        with self._lock:
            if key not in self._subjects or self._subjects[key].get("deleted"):
                return False
            new_subjects = dict(self._subjects)
            new_subjects[key] = {
                **self._subjects[key],
                "name": name,
                "notes": notes,
                "status": status,
            }
            self._write_json(new_subjects)
            self._subjects = new_subjects
            return True

    def subject_status(self, subject_id):
        with self._lock:
            subject = self._subjects.get(str(subject_id))
            if subject is None or subject.get("deleted"):
                return "Unclassified"
            status = subject.get("status", "Unclassified")
            return status if status in SUBJECT_STATUSES else "Unclassified"

    def subject_name(self, subject_id):
        with self._lock:
            subject = self._subjects.get(str(subject_id))
            if subject is None or subject.get("deleted"):
                return None
            return str(subject.get("name", f"Subject_{int(subject_id):03d}"))

    def delete_subject(self, subject_id):
        key = str(subject_id)
        with self._lock:
            if key not in self._subjects or self._subjects[key].get("deleted"):
                return False
            image_path = self.images_dir / f"subject_{int(key):03d}.jpg"
            try:
                image_path.unlink(missing_ok=True)
                for reference in self._subjects[key].get("references", []):
                    self._reference_path(subject_id, reference).unlink(missing_ok=True)
            except OSError as error:
                raise RuntimeError(f"Could not delete snapshot for Subject_{int(key):03d}.") from error
            new_subjects = dict(self._subjects)
            new_subjects[key] = {**self._subjects[key], "deleted": True}
            self._write_json(new_subjects)
            self._subjects = new_subjects
            return True

    def delete_all_subjects(self):
        with self._lock:
            active_keys = [
                key for key, subject in self._subjects.items()
                if not subject.get("deleted")
            ]
            for key in active_keys:
                image_path = self.images_dir / f"subject_{int(key):03d}.jpg"
                try:
                    image_path.unlink(missing_ok=True)
                    for reference in self._subjects[key].get("references", []):
                        self._reference_path(key, reference).unlink(missing_ok=True)
                except OSError as error:
                    raise RuntimeError(
                        f"Could not delete snapshot for Subject_{int(key):03d}."
                    ) from error

            if not active_keys:
                return 0

            new_subjects = dict(self._subjects)
            for key in active_keys:
                new_subjects[key] = {**self._subjects[key], "deleted": True}
            self._write_json(new_subjects)
            self._subjects = new_subjects
            return len(active_keys)

    def read_image(self, subject_id):
        key = str(subject_id)
        with self._lock:
            record = self._subjects.get(key)
            if record is None or record.get("deleted"):
                return None
            image_name = f"subject_{int(key):03d}.jpg"
            if record.get("image") != image_name:
                return None
            try:
                return (self.images_dir / image_name).read_bytes()
            except FileNotFoundError:
                return None

    def new_footage_paths(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"footage-{timestamp}-{secrets.token_hex(4)}.webm"
        return (
            self.footage_dir / filename.replace(".webm", "-partial.webm"),
            self.footage_dir / filename,
        )

    def list_footage(self):
        clips = []
        for path in self.footage_dir.iterdir():
            if path.is_file() and FOOTAGE_FILE_PATTERN.fullmatch(path.name):
                stat = path.stat()
                if stat.st_size:
                    clips.append({
                        "name": path.name,
                        "size": stat.st_size,
                        "modified": datetime.datetime.fromtimestamp(
                            stat.st_mtime
                        ).astimezone().strftime("%Y-%m-%d %H:%M"),
                    })
        return sorted(clips, key=lambda clip: clip["name"], reverse=True)

    def footage_path(self, filename):
        if not FOOTAGE_FILE_PATTERN.fullmatch(filename):
            return None
        path = self.footage_dir / filename
        return path if path.is_file() else None

    def delete_footage(self, filename):
        path = self.footage_path(filename)
        if path is None:
            return False
        try:
            path.unlink()
        except OSError as error:
            raise RuntimeError(f"Could not delete footage clip {filename}.") from error
        return True

    def delete_all_footage(self):
        deleted = 0
        for clip in self.list_footage():
            if self.delete_footage(clip["name"]):
                deleted += 1
        return deleted

    @staticmethod
    def _write_image(path, image_data):
        temporary_path = path.with_suffix(".jpg.tmp")
        try:
            temporary_path.write_bytes(image_data)
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _write_json(self, subjects):
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.data_dir,
                prefix="subjects-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(subjects, temporary_file, ensure_ascii=False, indent=2)
                temporary_file.write("\n")
            os.replace(temporary_path, self._file_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def _write_next_id(self, next_id):
        temporary_path = self._next_id_path.with_suffix(".tmp")
        try:
            temporary_path.write_text(str(next_id), encoding="ascii")
            os.replace(temporary_path, self._next_id_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def _configured_password():
    password = os.environ.get("FACE_TRACKER_PASSWORD")
    if password is None:
        return secrets.token_urlsafe(18)
    if len(password) < 12:
        raise RuntimeError(
            "FACE_TRACKER_PASSWORD must be at least 12 characters long."
        )
    return password


def _page(title, body):
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; font-family: Segoe UI, Arial, sans-serif; background: #fff; color: #18181b; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; background: #fff; }}
    .site-header {{ background: #5600d6; color: #fff; }}
    .header-content {{ width: min(1120px, calc(100% - 40px)); min-height: 86px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; gap: 20px; }}
    main {{ width: min(1120px, calc(100% - 40px)); margin: 0 auto; padding: 28px 0 48px; }}
    h1 {{ margin: 0; font-size: clamp(25px, 4vw, 32px); letter-spacing: -.02em; }}
    p {{ color: #52525b; line-height: 1.5; }}
    .count {{ color: #fff; font-size: 14px; }}
    .site-nav {{ display: flex; align-items: center; gap: 16px; }}
    .site-nav a {{ color: #fff; text-decoration: none; font-weight: 600; }}
    .site-nav a[aria-current="page"] {{ text-decoration: underline; text-underline-offset: 5px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 18px; }}
    .card, .login {{ background: #fff; border: 1px solid #e4e4e7; border-radius: 4px; overflow: hidden; }}
    .card img {{ display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: cover; background: #f4f4f5; }}
    .footage-card video {{ display: block; width: 100%; background: #18181b; }}
    .card-content {{ padding: 14px; }}
    .card-heading {{ display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }}
    .card-heading h2 {{ margin: 0; font-size: 16px; font-weight: 600; }}
    .subject-id {{ color: #6d28d9; font: 600 12px Segoe UI, Arial, sans-serif; white-space: nowrap; }}
    label {{ display: block; margin: 12px 0 6px; color: #3f3f46; font-size: 13px; }}
    input, textarea, select {{ width: 100%; border: 1px solid #d4d4d8; border-radius: 3px; padding: 10px 11px; color: #18181b; background: #fff; font: inherit; }}
    textarea {{ min-height: 68px; resize: vertical; }}
    button {{ border: 0; border-radius: 3px; padding: 10px 14px; background: #5600d6; color: #fff; font: inherit; font-weight: 600; cursor: pointer; }}
    button:hover {{ background: #4300ad; }}
    button:disabled {{ cursor: default; opacity: .55; }}
    .header-actions {{ display: flex; align-items: center; gap: 14px; }}
    .header-actions form {{ margin: 0; }}
    .remove-all {{ background: transparent; border: 1px solid rgba(255,255,255,.7); }}
    .remove-all:hover {{ background: rgba(255,255,255,.14); }}
    .sign-out {{ padding: 10px 12px; background: #fff; color: #5600d6; }}
    .sign-out:hover {{ background: #f1eafe; }}
    .edit {{ margin-top: 12px; }}
    .edit summary {{ color: #52525b; font-size: 13px; cursor: pointer; }}
    .delete-button {{ margin-top: 8px; background: #fff; border: 1px solid #d4d4d8; color: #3f3f46; }}
    .delete-button:hover {{ background: #f4f4f5; }}
    .login {{ max-width: 420px; padding: 26px; margin: 12vh auto; border-radius: 5px; }}
    .login h1 {{ margin-bottom: 20px; }}
    .empty {{ padding: 24px 0; color: #71717a; }}
    .clip-time {{ margin: 0 0 12px; color: #52525b; font-size: 13px; }}
    .clip-actions {{ display: flex; gap: 8px; }}
    .clip-actions a {{ color: #5600d6; align-self: center; font-size: 13px; }}
    @media (max-width: 620px) {{ .header-content {{ min-height: 72px; flex-wrap: wrap; padding: 14px 0; }} main {{ padding-top: 20px; }} .header-actions {{ gap: 10px; }} .count {{ display: none; }} }}
  </style>
</head>
<body>{body}</body>
</html>"""


def _login_page(error=False):
    message = '<p role="alert">Incorrect password.</p>' if error else ""
    body = f"""<section class="login">
  <h1>Subject archive</h1>
  {message}
  <form method="post" action="/login">
    <label for="password">Password</label>
    <input id="password" name="password" type="password" autocomplete="current-password" required autofocus>
    <button type="submit">Sign in</button>
  </form>
</section>"""
    return _page("Unlock subject archive", body)


def _archive_page(subjects, csrf_token):
    if subjects:
        cards = []
        for subject in subjects:
            subject_id = int(subject["subject_id"])
            name = html.escape(str(subject.get("name", f"Subject_{subject_id:03d}")), quote=True)
            notes = html.escape(str(subject.get("notes", "")))
            status = subject.get("status", "Unclassified")
            if status not in SUBJECT_STATUSES:
                status = "Unclassified"
            status_options = "".join(
                f'<option value="{option}"{" selected" if option == status else ""}>{option}</option>'
                for option in SUBJECT_STATUSES
            )
            cards.append(f"""<article class="card">
  <img src="/subjects/{subject_id}.jpg" alt="Snapshot of {name}">
  <div class="card-content">
    <div class="card-heading"><h2>{name}</h2><span class="subject-id">SUBJECT_{subject_id:03d}</span></div>
    <details class="edit"><summary>Edit profile</summary>
    <form method="post" action="/profile/{subject_id}">
      <input type="hidden" name="csrf_token" value="{html.escape(csrf_token, quote=True)}">
      <label for="name-{subject_id}">Profile name</label>
      <input id="name-{subject_id}" name="name" maxlength="60" value="{name}" required>
      <label for="notes-{subject_id}">Notes</label>
      <textarea id="notes-{subject_id}" name="notes" maxlength="500">{notes}</textarea>
      <label for="status-{subject_id}">Status</label>
      <select id="status-{subject_id}" name="status">{status_options}</select>
      <button type="submit">Save profile</button>
    </form>
    <form method="post" action="/subjects/{subject_id}/delete" data-confirm="Remove this subject?">
      <input type="hidden" name="csrf_token" value="{html.escape(csrf_token, quote=True)}">
      <button class="delete-button" type="submit">Remove subject</button>
    </form>
    </details>
  </div>
</article>""")
        archive_content = f'<div class="grid">{"".join(cards)}</div>'
    else:
        archive_content = '<div class="empty">No subjects</div>'

    body = f"""    <header class="site-header">
      <div class="header-content">
        <div class="header-actions">
      <nav class="site-nav" aria-label="Main navigation">
        <a href="/" aria-current="page">Subjects</a>
        <a href="/footage">Footage</a>
      </nav>
      <span class="count">{len(subjects)} {"subject" if len(subjects) == 1 else "subjects"}</span>
      <form method="post" action="/subjects/clear" data-confirm="Remove all subjects and snapshots?">
        <input type="hidden" name="csrf_token" value="{html.escape(csrf_token, quote=True)}">
        <button class="remove-all" type="submit"{" disabled" if not subjects else ""}>Remove all subjects</button>
      </form>
      <form method="post" action="/logout">
        <input type="hidden" name="csrf_token" value="{html.escape(csrf_token, quote=True)}">
        <button class="sign-out" type="submit">Sign out</button>
      </form>
    </div>
  </div>
</header>
<main id="archive" data-count="{len(subjects)}">
{archive_content}
</main>
<script src="/archive.js" defer></script>"""
    return _page("Subject profiles", body)


def _footage_page(clips, csrf_token):
    if clips:
        cards = []
        for clip in clips:
            filename = html.escape(clip["name"], quote=True)
            modified = html.escape(clip["modified"])
            clip_url = f"/footage/{filename}"
            cards.append(f"""<article class="card footage-card">
  <video controls preload="metadata" src="{clip_url}"></video>
  <div class="card-content">
    <p class="clip-time">{modified}</p>
    <div class="clip-actions">
      <a href="{clip_url}" download>Download</a>
      <form method="post" action="/footage/{filename}/delete" data-confirm="Remove this clip?">
        <input type="hidden" name="csrf_token" value="{html.escape(csrf_token, quote=True)}">
        <button class="delete-button" type="submit">Remove</button>
      </form>
    </div>
  </div>
</article>""")
        footage_content = f'<div class="grid">{"".join(cards)}</div>'
    else:
        footage_content = '<div class="empty">No footage yet</div>'

    body = f"""<header class="site-header">
  <div class="header-content">
    <nav class="site-nav" aria-label="Main navigation">
      <a href="/">Subjects</a>
      <a href="/footage" aria-current="page">Footage</a>
    </nav>
    <div class="header-actions">
      <span class="count">{len(clips)} {"clip" if len(clips) == 1 else "clips"}</span>
      <form method="post" action="/footage/clear" data-confirm="Remove all saved footage?">
        <input type="hidden" name="csrf_token" value="{html.escape(csrf_token, quote=True)}">
        <button class="remove-all" type="submit"{" disabled" if not clips else ""}>Remove all footage</button>
      </form>
      <form method="post" action="/logout">
        <input type="hidden" name="csrf_token" value="{html.escape(csrf_token, quote=True)}">
        <button class="sign-out" type="submit">Sign out</button>
      </form>
    </div>
  </div>
</header>
<main id="footage" data-count="{len(clips)}">
  <h1>Footage</h1>
  <p>Clips appear after faces leave the frame. Current recordings are not listed yet.</p>
  {footage_content}
</main>
<script src="/footage.js" defer></script>"""
    return _page("Footage", body)


def make_server(store, host=HOST, port=PORT, password=None):
    password = password if password is not None else _configured_password()
    if len(password) < 12:
        raise ValueError("The website password must be at least 12 characters long.")
    sessions = {}
    sessions_lock = threading.Lock()

    class RequestHandler(http.server.BaseHTTPRequestHandler):
        server_version = "LocalSubjectArchive"
        sys_version = ""

        def log_message(self, format_string, *args):
            return

        def _send(self, status, body=b"", content_type="text/html; charset=utf-8", headers=()):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'unsafe-inline'; img-src 'self'; "
                "form-action 'self'; frame-ancestors 'none'; base-uri 'none'",
            )
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _redirect(self, location, headers=()):
            self._send(303, b"", headers=(("Location", location), *headers))

        def _send_file(self, path):
            size = path.stat().st_size
            if size == 0:
                self._send(404, "Footage clip not found.")
                return
            start = 0
            end = size - 1
            status = 200
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
                if match is None or (not match.group(1) and not match.group(2)):
                    self._send(
                        416,
                        b"",
                        content_type="video/webm",
                        headers=(("Content-Range", f"bytes */{size}"),),
                    )
                    return
                if match.group(1):
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else end
                else:
                    suffix_length = int(match.group(2))
                    start = max(0, size - suffix_length)
                if start >= size or end < start:
                    self._send(
                        416,
                        b"",
                        content_type="video/webm",
                        headers=(("Content-Range", f"bytes */{size}"),),
                    )
                    return
                end = min(end, size - 1)
                status = 206

            with path.open("rb") as footage_file:
                footage_file.seek(start)
                remaining = end - start + 1
                self.send_response(status)
                self.send_header("Content-Type", "video/webm")
                self.send_header("Content-Length", str(remaining))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self'; style-src 'unsafe-inline'; "
                    "img-src 'self'; media-src 'self'; form-action 'self'; "
                    "frame-ancestors 'none'; base-uri 'none'",
                )
                self.send_header("Accept-Ranges", "bytes")
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if self.command != "HEAD":
                    while remaining:
                        chunk = footage_file.read(min(64 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)

        def _local_request(self):
            try:
                client_is_local = ipaddress.ip_address(self.client_address[0]).is_loopback
                parsed_host = urlsplit("//" + self.headers.get("Host", ""))
                host_is_local = (
                    parsed_host.hostname in ("127.0.0.1", "localhost")
                    and parsed_host.port == self.server.server_port
                )
            except ValueError:
                client_is_local = False
                host_is_local = False
            if not client_is_local or not host_is_local:
                self._send(403, "This archive is only available from this computer.")
                return False
            return True

        def _session(self):
            try:
                cookies = http.cookies.SimpleCookie(self.headers.get("Cookie", ""))
            except http.cookies.CookieError:
                return None, None
            morsel = cookies.get("subject_archive")
            if morsel is None:
                return None, None
            with sessions_lock:
                csrf_token = sessions.get(morsel.value)
            return (morsel.value, csrf_token) if csrf_token else (None, None)

        def _read_form(self):
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send(400, "Invalid form submission.")
                return None
            if content_length < 0 or content_length > MAX_FORM_BYTES:
                self._send(413, "Form submission is too large.")
                return None
            body = self.rfile.read(content_length)
            return parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)

        def do_GET(self):
            if not self._local_request():
                return
            path = urlsplit(self.path).path
            if path == "/":
                _, csrf_token = self._session()
                if csrf_token:
                    self._send(200, _archive_page(store.list_subjects(), csrf_token))
                else:
                    self._send(200, _login_page())
                return

            _, csrf_token = self._session()
            if csrf_token is None:
                self._redirect("/")
                return

            if path == "/archive.js":
                self._send(200, ARCHIVE_SCRIPT, content_type="text/javascript; charset=utf-8")
                return
            if path == "/footage":
                self._send(200, _footage_page(store.list_footage(), csrf_token))
                return
            if path == "/footage.js":
                self._send(200, FOOTAGE_SCRIPT, content_type="text/javascript; charset=utf-8")
                return
            if path == "/api/count":
                self._send(
                    200,
                    json.dumps({"count": len(store.list_subjects())}),
                    content_type="application/json; charset=utf-8",
                )
                return
            if path == "/api/footage-count":
                self._send(
                    200,
                    json.dumps({"count": len(store.list_footage())}),
                    content_type="application/json; charset=utf-8",
                )
                return

            match = re.fullmatch(r"/subjects/([1-9][0-9]*)\.jpg", path)
            if match:
                image = store.read_image(int(match.group(1)))
                if image is None:
                    self._send(404, "Snapshot not found.")
                else:
                    self._send(200, image, content_type="image/jpeg")
                return

            match = re.fullmatch(r"/footage/(footage-[0-9]{8}-[0-9]{6}-[0-9a-f]{8}\.webm)", path)
            if match:
                footage_path = store.footage_path(match.group(1))
                if footage_path is None:
                    self._send(404, "Footage clip not found.")
                else:
                    self._send_file(footage_path)
                return

            if path == "/favicon.ico":
                self._send(204)
                return
            self._send(404, "Not found.")

        def do_POST(self):
            if not self._local_request():
                return
            path = urlsplit(self.path).path
            form = self._read_form()
            if form is None:
                return

            if path == "/login":
                supplied_password = form.get("password", [""])[0]
                if secrets.compare_digest(supplied_password, password):
                    session_id = secrets.token_urlsafe(32)
                    csrf_token = secrets.token_urlsafe(32)
                    with sessions_lock:
                        sessions[session_id] = csrf_token
                    cookie = (
                        f"subject_archive={session_id}; Path=/; HttpOnly; SameSite=Strict"
                    )
                    self._redirect("/", headers=(("Set-Cookie", cookie),))
                else:
                    self._send(401, _login_page(error=True))
                return

            session_id, csrf_token = self._session()
            if csrf_token is None:
                self._redirect("/")
                return
            supplied_token = form.get("csrf_token", [""])[0]
            if not secrets.compare_digest(supplied_token, csrf_token):
                self._send(403, "Invalid form token. Reload the page and try again.")
                return

            if path == "/logout":
                with sessions_lock:
                    sessions.pop(session_id, None)
                cookie = "subject_archive=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"
                self._redirect("/", headers=(("Set-Cookie", cookie),))
                return

            match = re.fullmatch(r"/profile/([1-9][0-9]*)", path)
            if match:
                name = form.get("name", [""])[0].strip()
                notes = form.get("notes", [""])[0].strip()
                status = form.get("status", ["Unclassified"])[0]
                if (
                    not name or len(name) > 60 or len(notes) > 500
                    or status not in SUBJECT_STATUSES
                ):
                    self._send(400, "Invalid profile.")
                    return
                if not store.update_profile(int(match.group(1)), name, notes, status):
                    self._send(404, "Subject not found.")
                    return
                self._redirect("/")
                return

            match = re.fullmatch(r"/subjects/([1-9][0-9]*)/delete", path)
            if match:
                if not store.delete_subject(int(match.group(1))):
                    self._send(404, "Subject not found.")
                    return
                self._redirect("/")
                return

            if path == "/subjects/clear":
                store.delete_all_subjects()
                self._redirect("/")
                return

            match = re.fullmatch(
                r"/footage/(footage-[0-9]{8}-[0-9]{6}-[0-9a-f]{8}\.webm)/delete",
                path,
            )
            if match:
                if not store.delete_footage(match.group(1)):
                    self._send(404, "Footage clip not found.")
                    return
                self._redirect("/footage")
                return

            if path == "/footage/clear":
                store.delete_all_footage()
                self._redirect("/footage")
                return

            self._send(404, "Not found.")

        def do_HEAD(self):
            self.do_GET()

    server = http.server.ThreadingHTTPServer((host, port), RequestHandler)
    server.daemon_threads = True
    server.allow_reuse_address = True
    return server


def start_site(store):
    configured_password = os.environ.get("FACE_TRACKER_PASSWORD")
    password = _configured_password()
    server = make_server(store, password=password)
    thread = threading.Thread(target=server.serve_forever, name="local-subject-site", daemon=True)
    thread.start()
    url = f"http://{HOST}:{PORT}/"
    if configured_password is None:
        print(f"Temporary archive password: {password}")
    else:
        print("Archive password loaded from FACE_TRACKER_PASSWORD.")
    try:
        browser_opened = webbrowser.open(url)
    except (OSError, webbrowser.Error) as error:
        print(f"Could not open a browser ({error}). Open the subject archive at {url}")
    else:
        if not browser_opened:
            print(f"Open the subject archive at {url}")
    return server


def stop_site(server):
    server.shutdown()
    server.server_close()
