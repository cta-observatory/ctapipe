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
def server(server_tmp_path):
    s = Server(server_tmp_path)
    s.start()
    try:
        yield s
    finally:
        s.event.set()
        s.join()
