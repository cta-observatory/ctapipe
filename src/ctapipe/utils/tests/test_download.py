import os
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Event, Thread

import pytest


class Server(Thread):
    def __init__(self, directory):
        super().__init__()
        self.event = Event()
        handler = partial(SimpleHTTPRequestHandler, directory=directory)
        self.httpd = HTTPServer(("", 0), handler)
        self.url = f"http://localhost:{self.httpd.server_port}"
        self.httpd.timeout = 0.1
        self.httpd.handle_timeout = lambda: None

    def run(self):
        with self.httpd:
            while not self.event.is_set():
                self.httpd.handle_request()


@pytest.fixture(scope="module")
def server_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("server")


@pytest.fixture(scope="module")
def test_file(server_tmp_path):
    path = server_tmp_path / "foo.csv"
    with path.open("w") as f:
        f.write("foo,bar\n1,2\n3,4")
    return path


@pytest.fixture(scope="module")
def server(server_tmp_path):
    s = Server(server_tmp_path)
    s.start()
    try:
        yield s
    finally:
        s.event.set()
        s.join()


def test_download_file(server, test_file, tmp_path):
    from ctapipe.utils.download import download_file

    path = tmp_path / test_file.name
    download_file(f"{server.url}/{test_file.name}", path)

    with path.open() as downloaded, test_file.open() as original:
        assert downloaded.read() == original.read()


def test_download_file_cached(server, test_file, tmp_path):
    from ctapipe.utils.download import download_file_cached

    before = os.getenv("CTAPIPE_CACHE")

    try:
        os.environ["CTAPIPE_CACHE"] = str(tmp_path)
        path = download_file_cached(
            test_file.name, cache_name="tests", default_url=server.url
        )

        with path.open() as downloaded, test_file.open() as original:
            assert downloaded.read() == original.read()
    finally:
        if before is None:
            del os.environ["CTAPIPE_CACHE"]
        else:
            os.environ["CTAPIPE_CACHE"] = before
