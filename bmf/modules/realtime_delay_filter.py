from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, \
    VideoFrame, AudioFrame # 导入 VideoFrame 和 AudioFrame 只是为了示例，实际逻辑中不直接使用
import time
import sys

class RealtimeDelayFilter(Module):
    '''
    BMF Python 模块，模拟 FFmpeg 的 -re 功能，
    根据数据包的时间戳控制发送速率，实现实时读取。
    '''
    def __init__(self, node, option=None):
        '''
        模块初始化方法
        Args:
            node: BMF 节点对象
            option: 模块选项，可以通过图配置传递参数
        '''
        self.node_ = node
        self.option_ = option
        Log.log_node(LogLevel.INFO, self.node_, f"RealtimeReadModule initialized with option: {option}")

        # 初始化时间相关变量
        self._start_real_time = None # 记录处理第一个数据包时的实际系统时间
        self._first_packet_timestamp = Timestamp.UNSET # 记录第一个数据包的时间戳 (微秒单位)

    def process(self, task):
        '''
        模块处理方法，BMF 引擎会周期性调用此方法来处理输入数据
        Args:
            task: BMF 任务对象，包含输入和输出队列
        Returns:
            ProcessResult: 模块处理结果状态
        '''
        # 遍历任务的所有输入队列
        # 对于模拟 -re 的模块，通常只有一个输入流
        for (input_id, input_packets) in task.get_inputs().items():

            # 获取与输入队列对应的输出队列
            output_packets = task.get_outputs()[input_id]

            # 处理输入队列中的所有数据包
            while not input_packets.empty():
                # 从输入队列获取一个数据包
                pkt = input_packets.get()

                # 处理 EOS (End of Stream) 数据包
                if pkt.timestamp == Timestamp.EOF:
                    Log.log_node(LogLevel.DEBUG, self.node_, f"Receive EOF on input {input_id}")
                    # 将 EOS 包发送到输出队列
                    output_packets.put(Packet.generate_eof_packet())
                    # 设置任务状态为 DONE，通知 BMF 引擎此任务已完成
                    task.timestamp = Timestamp.DONE
                    # 返回 OK，表示任务处理顺利结束
                    return ProcessResult.OK

                # 处理有效的数据包 (非 EOF 且时间戳有效)
                if pkt.defined() and pkt.timestamp != Timestamp.UNSET:
                    current_pkt_timestamp = pkt.timestamp # 数据包的时间戳，单位为微秒

                    # 如果是处理的第一个有效数据包，记录开始时间和时间戳
                    if self._first_packet_timestamp == Timestamp.UNSET:
                        self._start_real_time = time.time() # 记录实际开始处理的时间 (秒)
                        self._first_packet_timestamp = current_pkt_timestamp
                        Log.log_node(LogLevel.INFO, self.node_, f"First packet timestamp: {self._first_packet_timestamp} us, Start real time: {self._start_real_time:.4f} s")

                    # 计算数据包相对于第一个包的时间差（微秒）
                    pkt_time_diff_us = current_pkt_timestamp - self._first_packet_timestamp

                    # 计算数据包相对于开始时间点理论上应该发送的实际时间（秒）
                    # 将微秒转换为秒：除以 1,000,000
                    expected_real_time_sec = self._start_real_time + pkt_time_diff_us / 1000000.0

                    # 获取当前的实际系统时间（秒）
                    current_real_time_sec = time.time()

                    # 计算需要等待的时间（秒）
                    # 如果理论发送时间在未来，则需要等待
                    wait_time_sec = expected_real_time_sec - current_real_time_sec

                    if wait_time_sec > 0:
                        # 进行等待，模拟实时播放的延迟
                        # Log.log_node(LogLevel.DEBUG, self.node_, f"Waiting for {wait_time_sec:.4f} seconds for packet with timestamp {current_pkt_timestamp} us")
                        time.sleep(wait_time_sec)

                    # 将数据包放入输出队列，发送给下游模块
                    output_packets.put(pkt)

                # 对于无效的数据包（未定义或时间戳为 UNSET），直接忽略或根据需求处理
                # 在这个简单实现中，我们只处理有效数据包和 EOF

        # 如果循环结束，输入队列已空，但没有收到 EOS，说明当前没有更多数据可处理，
        # 或者正在等待合适的发送时间。返回 AGAIN 通知 BMF 引擎稍后再次调用 process 方法。
        return ProcessResult.OK
