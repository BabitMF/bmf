from bmf import *
from bmf.lib._bmf import sdk
import numpy as np
import bmf.hml.hmp as mp
import librosa

if sys.version_info.major == 2:
    from Queue import Queue
else:
    from queue import Queue

class BMFRobotEffect(Module):
    def __init__(self, node, option=None):
        self.eof_received_ = False
        self.is_first_frame = True
        self._audio_frame_cache = Queue()
        self._input_frame_pts = Queue()
        self._input_frame_nsamples = []
        self._sample_rate = None
        self._time_base = None
        self._layout = None

    def process(self, task):
        output_packets = None
        for input_id, input_packets in task.get_inputs().items():
            output_packets = task.get_outputs()[input_id]
            pkt = input_packets.get()
            
            if pkt.timestamp == Timestamp.EOF:
                self.eof_received_ = True
                break 

            if pkt.is_(AudioFrame):
                in_frame = pkt.get(AudioFrame)
                if self.is_first_frame:
                    self.validate_audio_format(in_frame)
                    self._sample_rate = in_frame.sample_rate
                    self._time_base = in_frame.time_base
                    self._layout = in_frame.layout
                    self.is_first_frame = False

                if in_frame and pkt.timestamp != Timestamp.UNSET:
                    self._input_frame_pts.put((in_frame.pts, pkt.timestamp))
                    self._audio_frame_cache.put(in_frame)   
                    self._input_frame_nsamples.append(in_frame.nsamples)
                 
        if self.eof_received_:
            Log.log_node(LogLevel.DEBUG, task.get_node(), "Receive EOF")
            if self._audio_frame_cache and output_packets:
                self.process_and_output_frame(output_packets)
                output_packets.put(Packet.generate_eof_packet())
            task.set_timestamp(Timestamp.DONE)
        return ProcessResult.OK

    def robot_effect(self, buffer, mod_frequency = 15.0, pitch_shift = -12, echo_strength = 0.3):
        audio_data = np.mean(buffer, axis=0) if buffer.shape[0] == 2 else buffer[0]

        # Ampltitude Modulation
        t = np.linspace(0, len(audio_data) / self._sample_rate, len(audio_data), endpoint=False)
        lfo = np.sin(2 * np.pi * mod_frequency * t)
        modulated_audio = audio_data * lfo

        # Pitch Shifting and Distortion
        shifted_audio = librosa.effects.pitch_shift(modulated_audio, sr = self._sample_rate, n_steps = pitch_shift)
        distorted_audio = np.clip(shifted_audio, -0.5, 0.5)

        # Echo Effect
        delay_samples = int(self._sample_rate * 0.1)  # 100ms delay for echo
        echo = np.pad(distorted_audio, (delay_samples, 0), mode='constant')[:len(distorted_audio)]
        robotic_audio = distorted_audio + echo_strength * echo

        # Normalize and Reshape to Stereo if Necessary
        robotic_audio = robotic_audio / np.max(np.abs(robotic_audio))
        if buffer.shape[0] == 2:
            robotic_audio = np.array([robotic_audio, robotic_audio])
        
        print(f"output shape is {robotic_audio.shape}")
        return robotic_audio

    def merge_audio_frames(self):
        data = []
        while not self._audio_frame_cache.empty():
            in_frame = self._audio_frame_cache.get()
            frame_array = self.get_audioframe(in_frame)
            data.append(frame_array)
        data = np.concatenate(data, axis = -1)
        print(f"Accumulated buffer shape: {data.shape}")
        return data
    
    def send_audio_frames(self, processed_buffer,output_packets):
        split_data = np.split(processed_buffer, np.cumsum(self._input_frame_nsamples)[:-1], axis=1)
        for data in split_data:
            if self._input_frame_pts.empty():
                 raise ValueError("Error: _input_frame_pts queue is empty when attempting to get frame_pts and packet_timestamp.")
            frame_pts, packet_timestamp = self._input_frame_pts.get()
            in_data = [mp.from_numpy(m) for m in data]
            out_frame = AudioFrame(in_data, self._layout, True)
            out_frame.sample_rate = self._sample_rate
            out_frame.pts = frame_pts
            out_frame.time_base = self._time_base

            if output_packets is not None:
                output_pkt = Packet(out_frame)
                output_pkt.timestamp = packet_timestamp
                output_packets.put(output_pkt)

    def process_and_output_frame(self, output_packets):
        buffer = self.merge_audio_frames()
        processed_buffer = self.robot_effect(buffer)
        self.send_audio_frames(processed_buffer, output_packets)

    def validate_audio_format(self, in_frame):
        if not (8000 <= in_frame.sample_rate <= 96000):
            raise ValueError(f"Invalid sample rate: {in_frame.sample_rate}. Must be between 8000 and 96000.")
    
        if in_frame.layout not in [sdk.kLAYOUT_MONO, sdk.kLAYOUT_STEREO]:
            raise ValueError(f"Invalid layout: {in_frame.layout}. Supported layouts are MONO and STEREO.")
    
        if in_frame.dtype not in [mp.kFloat32, mp.kInt16]:
            raise ValueError(f"Invalid data type: {in_frame.dtype}. Supported types are Float32 and Int16.")
    
    def get_audioframe(self, in_frame):
        frame_array = []
        planes_np = mp.to_numpy(in_frame.planes)

        if in_frame.layout == sdk.kLAYOUT_MONO: 
            frame_array = planes_np[0][:in_frame.nsamples]
            frame_array = np.reshape(frame_array, (1, -1)) # (1, T)
        
        elif in_frame.layout == sdk.kLAYOUT_STEREO:
            if in_frame.planer:
                frame_array = np.array([plane[:in_frame.nsamples] for plane in planes_np])  # (2, T)
            else:
                frame_array = planes_np[0][:in_frame.nsamples * 2]  # (T * 2)
                frame_array = frame_array.reshape([-1, 2]).T  # (2, T)
        
        if in_frame.dtype == mp.kInt16:
            frame_array = (frame_array / 32767.0).astype(np.float32)
        
        return frame_array
    

    
    