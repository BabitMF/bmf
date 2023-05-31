#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
"""
Very simple HTTP server in python for logging requests
Usage::
    ./server.py [<port>]
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
from http.server import HTTPServer

import logging
from queue import Queue
import threading
import json


class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_GET(self):
        # logging.info(
        #    "GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers)
        # )
        self._set_response()
        request = {"method": "inspect", "path": self.path}
        self.server.queue.put(json.dumps(request))
        item = self.server.recv_queue.get()
        print("get item: ", item)
        self.wfile.write(
            "GET request for {}\ninspect:\n{}\n".format(self.path, item).encode("utf-8")
        )

    def do_POST(self):
        content_length = int(
            self.headers["Content-Length"]
        )  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        logging.info(
            "POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
            str(self.path),
            str(self.headers),
            post_data.decode("utf-8"),
        )

        self.server.queue.put(post_data)

        self._set_response()
        self.wfile.write("POST request {} done".format(post_data).encode("utf-8"))


class BroadcasterHTTPServer(HTTPServer):
    """this class is necessary to allow passing custom request handler into
    the RequestHandlerClass"""

    def __init__(self, server_address, RequestHandlerClass, queue, recv_queue):
        HTTPServer.__init__(self, server_address, RequestHandlerClass)
        self.queue = queue
        self.recv_queue = recv_queue


def run(rpc_queue, recv_queue, port=55566):
    logging.basicConfig(level=logging.INFO)
    server_address = ("", port)
    httpd = BroadcasterHTTPServer(server_address, S, rpc_queue, recv_queue)
    logging.info("Starting httpd...\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info("Stopping httpd...\n")


def process(queue):
    while True:
        item = queue.get()
        logging.info("get item: %s\n", item)


if __name__ == "__main__":
    from sys import argv

    queue = Queue(1)
    recv_queue = Queue(1)
    t = threading.Thread(
        target=process,
        args=(queue,),
    )
    t.start()
    if len(argv) == 2:
        run(queue, port=int(argv[1]))
    else:
        run(queue)
    t.join()
