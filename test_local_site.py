import http.client
from html.parser import HTMLParser
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from local_site import SubjectStore, _configured_password, make_server


class HiddenFieldParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden_fields = {}

    def handle_starttag(self, tag, attrs):
        if tag == "input":
            attributes = dict(attrs)
            if attributes.get("type") == "hidden":
                self.hidden_fields[attributes.get("name")] = attributes.get("value")


class LocalSiteTests(unittest.TestCase):
    PASSWORD = "test-password-123"

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = SubjectStore(self.temp_dir.name)
        self.store.add_subject(1, b"test jpeg bytes")
        self.server = make_server(self.store, port=0, password=self.PASSWORD)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=3
        )

    def tearDown(self):
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.temp_dir.cleanup()

    def request(self, method, path, body=None, headers=None):
        self.connection.request(method, path, body=body, headers=headers or {})
        response = self.connection.getresponse()
        return response.status, response.read(), response.getheaders()

    def login(self):
        status, _, headers = self.request(
            "POST",
            "/login",
            urlencode({"password": self.PASSWORD}),
            {"Content-Type": "application/x-www-form-urlencoded"},
        )
        self.assertEqual(status, 303)
        return dict(headers)["Set-Cookie"].split(";", 1)[0]

    def test_login_profile_edit_and_logout(self):
        status, body, _ = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Sign in", body)

        cookie = self.login()
        status, body, _ = self.request("GET", "/api/count", headers={"Cookie": cookie})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"count": 1})

        status, body, _ = self.request("GET", "/archive.js", headers={"Cookie": cookie})
        self.assertEqual(status, 200)
        self.assertIn(b"checkForNewSubjects", body)

        status, body, _ = self.request("GET", "/", headers={"Cookie": cookie})
        self.assertEqual(status, 200)
        self.assertIn(b"Subject_001", body)
        self.assertIn(b"Remove all subjects", body)
        self.assertIn(b'href="/footage">Footage</a>', body)
        self.assertIn(b"color-scheme: light", body)
        self.assertNotIn(b"<h1>Subject archive</h1>", body)
        for label in (b"Suspicious", b"Recognized", b"Unclassified"):
            self.assertIn(label, body)

        parser = HiddenFieldParser()
        parser.feed(body.decode("utf-8"))
        csrf_token = parser.hidden_fields["csrf_token"]
        status, _, _ = self.request(
            "POST",
            "/profile/1",
            urlencode({
                "csrf_token": csrf_token,
                "name": "Test subject",
                "notes": "Local profile",
                "status": "Recognized",
            }),
            {
                "Cookie": cookie,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        self.assertEqual(status, 303)
        self.assertEqual(self.store.list_subjects()[0]["name"], "Test subject")
        self.assertEqual(self.store.subject_status(1), "Recognized")
        status, body, _ = self.request("GET", "/", headers={"Cookie": cookie})
        self.assertEqual(status, 200)
        self.assertIn(b'<option value="Recognized" selected>', body)
        self.assertIn(b"Segoe UI, Arial, sans-serif", body)

        status, body, _ = self.request("GET", "/", headers={"Cookie": cookie})
        parser = HiddenFieldParser()
        parser.feed(body.decode("utf-8"))
        status, _, _ = self.request(
            "POST",
            "/logout",
            urlencode({"csrf_token": parser.hidden_fields["csrf_token"]}),
            {
                "Cookie": cookie,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        self.assertEqual(status, 303)
        status, body, _ = self.request("GET", "/", headers={"Cookie": cookie})
        self.assertEqual(status, 200)
        self.assertIn(b"Sign in", body)

    def test_profile_rejects_unknown_status(self):
        cookie = self.login()
        status, body, _ = self.request("GET", "/", headers={"Cookie": cookie})
        parser = HiddenFieldParser()
        parser.feed(body.decode("utf-8"))
        csrf_token = parser.hidden_fields["csrf_token"]
        status, body, _ = self.request(
            "POST",
            "/profile/1",
            urlencode({
                "csrf_token": csrf_token,
                "name": "Test subject",
                "notes": "",
                "status": "Unsafe",
            }),
            {
                "Cookie": cookie,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        self.assertEqual(status, 400)
        self.assertEqual(self.store.subject_status(1), "Unclassified")

    def test_rejects_non_local_host_header(self):
        status, body, _ = self.request("GET", "/", headers={"Host": "example.com"})
        self.assertEqual(status, 403)
        self.assertIn(b"only available from this computer", body)

    def test_generated_and_configured_passwords(self):
        with patch("local_site.os.environ.get", return_value=None):
            self.assertGreaterEqual(len(_configured_password()), 20)
        with patch(
            "local_site.os.environ.get",
            return_value=self.PASSWORD,
        ):
            self.assertEqual(_configured_password(), self.PASSWORD)
        with patch("local_site.os.environ.get", return_value="short"):
            with self.assertRaisesRegex(RuntimeError, "at least 12 characters"):
                _configured_password()

    def test_subject_data_persists_and_next_id_is_not_reused(self):
        self.assertEqual(self.store.next_subject_id(), 2)
        self.store.delete_subject(1)
        reopened_store = SubjectStore(self.temp_dir.name)
        self.assertEqual(reopened_store.next_subject_id(), 2)
        self.assertIsNone(reopened_store.read_image(1))
        self.assertTrue(reopened_store.has_subject(1))
        self.assertEqual(reopened_store.list_subjects(), [])

    def test_face_references_persist_with_the_subject_and_are_removed(self):
        self.assertFalse(self.store.add_reference_image(1, b"test jpeg bytes"))
        self.assertTrue(self.store.add_reference_image(1, b"reference image"))
        self.assertEqual(
            self.store.read_reference_images(1),
            [b"reference image"],
        )
        reopened_store = SubjectStore(self.temp_dir.name)
        self.assertEqual(
            reopened_store.read_reference_images(1),
            [b"reference image"],
        )
        self.assertFalse(self.store.add_reference_image(1, b"reference image"))
        self.store.delete_subject(1)
        self.assertEqual(self.store.read_reference_images(1), [])

    def test_footage_page_streaming_and_bulk_removal(self):
        _, clip_path = self.store.new_footage_paths()
        clip_path.write_bytes(b"0123456789")
        cookie = self.login()

        status, body, _ = self.request(
            "GET",
            "/footage",
            headers={"Cookie": cookie},
        )
        self.assertEqual(status, 200)
        self.assertIn(b"<h1>Footage</h1>", body)
        self.assertIn(b"Current recordings are not listed yet.", body)
        self.assertIn(b"Remove all footage", body)
        self.assertIn(clip_path.name.encode(), body)

        status, _, headers = self.request(
            "GET",
            f"/footage/{clip_path.name}",
        )
        self.assertEqual(status, 303)
        self.assertEqual(dict(headers)["Location"], "/")

        status, body, headers = self.request(
            "GET",
            f"/footage/{clip_path.name}",
            headers={"Cookie": cookie, "Range": "bytes=2-5"},
        )
        self.assertEqual(status, 206)
        self.assertEqual(body, b"2345")
        self.assertEqual(dict(headers)["Content-Range"], "bytes 2-5/10")

        status, body, _ = self.request(
            "GET",
            "/",
            headers={"Cookie": cookie},
        )
        parser = HiddenFieldParser()
        parser.feed(body.decode("utf-8"))
        status, _, _ = self.request(
            "POST",
            "/footage/clear",
            urlencode({"csrf_token": parser.hidden_fields["csrf_token"]}),
            {
                "Cookie": cookie,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        self.assertEqual(status, 303)
        self.assertEqual(self.store.list_footage(), [])

    def test_delete_route_removes_saved_subject(self):
        cookie = self.login()
        status, body, _ = self.request("GET", "/", headers={"Cookie": cookie})
        parser = HiddenFieldParser()
        parser.feed(body.decode("utf-8"))
        csrf_token = parser.hidden_fields["csrf_token"]
        status, _, _ = self.request(
            "POST",
            "/subjects/1/delete",
            urlencode({"csrf_token": csrf_token}),
            {
                "Cookie": cookie,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        self.assertEqual(status, 303)
        self.assertEqual(self.store.list_subjects(), [])

    def test_remove_all_route_clears_archive_but_never_reuses_ids(self):
        self.store.add_subject(2, b"second test jpeg")
        cookie = self.login()
        status, body, _ = self.request("GET", "/", headers={"Cookie": cookie})
        parser = HiddenFieldParser()
        parser.feed(body.decode("utf-8"))
        csrf_token = parser.hidden_fields["csrf_token"]
        status, _, _ = self.request(
            "POST",
            "/subjects/clear",
            urlencode({"csrf_token": csrf_token}),
            {
                "Cookie": cookie,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        self.assertEqual(status, 303)
        self.assertEqual(self.store.list_subjects(), [])
        self.assertEqual(self.store.next_subject_id(), 3)
        self.assertIsNone(self.store.read_image(1))
        self.assertIsNone(self.store.read_image(2))


if __name__ == "__main__":
    unittest.main()
