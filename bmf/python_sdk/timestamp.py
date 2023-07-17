import sys


class Timestamp:
    UNSET = -1
    PAUSE = sys.maxsize - 5  # indicate this is a PAUSE packet
    DYN_EOS = sys.maxsize - 4  # indicate this is a dynamical update caused EOS packet
    EOF = sys.maxsize - 3  # indicate this is a EOF packet
    EOS = sys.maxsize - 2  # indicate this is a EOS packet
    INF_SRC = sys.maxsize - 1  # indicate this is a task of source node
    DONE = sys.maxsize  # indicate this stream is done
