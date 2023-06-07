from bmf import Log, LogLevel
from enum import Enum
import bisect


class KeyWrapper:
    def __init__(self, iterable, key):
        self.it = iterable
        self.key = key

    def __getitem__(self, i):
        return self.key(self.it[i])

    def __len__(self):
        return len(self.it)


class BufferStates(Enum):
    STREAMING = 0
    BUFFERING = 1


class JitterBuffer:
    def __init__(self, jid, is_audio, rebind_callback):
        self.last_target_timestamp = None
        self.last_get_pkt_timestamp = None
        self.is_audio = is_audio
        self.last_pkt = None
        self.pkt_list = []
        self.get_eof_pkt = False
        # offset between origin pts with new pts in new timeline
        self.offset = None
        self.jid = jid
        self.is_audio = is_audio
        self.buffer_state = BufferStates.BUFFERING
        self.rebind_callback = rebind_callback
        # in microseconds
        self.low_water_level = 400000
        self.balanced_water_level = 600000
        self.hight_water_level = 800000

    def get_offset(self):
        return self.offset

    def set_offset(self, offset):
        self.offset = offset

    def push_packet(self, pkt):
        if self.offset is not None:
            pkt.timestamp = pkt.timestamp + self.offset
        self.pkt_list.append(pkt)

    def get_packet(self, target_timestamp):
        Log.log(
            LogLevel.DEBUG,
            "jid: ",
            self.jid,
            " target timestamp: ",
            target_timestamp,
        )

        if self.buffer_state == BufferStates.BUFFERING:
            # last pkt
            Log.log(
                LogLevel.DEBUG,
                "buffering state jid: ",
                self.jid,
                " cache_time: ",
                self.cache_time(),
            )

            if self.cache_time() > self.balanced_water_level or self.is_get_eof_pkt():
                # change state and rebind timestamp
                self.buffer_state = BufferStates.STREAMING
                self.rebind_timestamp(target_timestamp)
            return None if self.is_audio else self.last_pkt

        elif self.buffer_state == BufferStates.STREAMING:
            Log.log(
                LogLevel.DEBUG,
                "streaming state jid: ",
                self.jid,
                " cache_time: ",
                self.cache_time(),
            )
            if not self.pkt_list:
                Log.log(
                    LogLevel.WARNING,
                    "streaming state jid: ",
                    self.jid,
                    "pkt list is empty",
                )
                self.last_target_timestamp = target_timestamp
                return None if self.is_audio else self.last_pkt

            target_pkt = None

            # get a pkt
            index = bisect.bisect_right(
                KeyWrapper(self.pkt_list, key=lambda pkt: pkt.timestamp),
                target_timestamp,
            )
            if index == 0:
                target_pkt = self.pkt_list[0]
                Log.log(
                    LogLevel.DEBUG,
                    "jid: ",
                    self.jid,
                    " target timestamp is less than first pkt timestamp: ",
                    target_pkt.timestamp,
                )
            else:
                target_pkt = self.pkt_list[index - 1]
                for _ in range(index):
                    self.pkt_list.pop(0)

                Log.log(
                    LogLevel.DEBUG,
                    "jid: ",
                    self.jid,
                    " get packet, target index in cache list: ",
                    index,
                    "target pkt timestamp: ",
                    target_pkt.timestamp,
                )

            self.last_pkt = target_pkt
            self.last_get_pkt_timestamp = self.last_pkt.timestamp
            self.last_target_timestamp = target_timestamp
            if self.cache_time() < self.low_water_level and not self.is_get_eof_pkt():
                self.buffer_state = BufferStates.BUFFERING
            return target_pkt

    def cache_time(self):
        if not self.pkt_list:
            return 0
        pkt_start = self.pkt_list[0]
        pkt_end = self.pkt_list[-1]
        return pkt_end.timestamp - pkt_start.timestamp

    def is_empty(self):
        return len(self.pkt_list) == 0

    def add_timestamp_offset(self, offset):
        for pkt in self.pkt_list:
            t = pkt.timestamp
            pkt.timestamp = t + offset

        # record offset
        Log.log(LogLevel.INFO, "add offset: ", offset)

    def rebind_timestamp(self, target_timestamp):
        if not self.pkt_list:
            Log.log(
                LogLevel.WARNING,
                "jid: ",
                self.jid,
                " can not rebind when there is no pkt",
            )
            return

        Log.log(
            LogLevel.DEBUG,
            "jid: ",
            self.jid,
            "rebind timestamp, old offset: ",
            self.offset,
            " target_timestamp: ",
            target_timestamp,
            " pkt[0] timestamp: ",
            self.pkt_list[0].timestamp,
        )
        # bind pkt[0].timestamp with target_timestamp
        # set the right offset of origin pkt timestamp
        if self.offset is None:
            self.offset = target_timestamp - self.pkt_list[0].timestamp
        else:
            self.offset = target_timestamp - self.pkt_list[0].timestamp + self.offset

        offset = target_timestamp - self.pkt_list[0].timestamp
        self.add_timestamp_offset(offset)

        # callback
        self.rebind_callback(self.jid, self.offset)

    def set_eof_get(self):
        self.get_eof_pkt = True

    def is_get_eof_pkt(self):
        return self.get_eof_pkt
