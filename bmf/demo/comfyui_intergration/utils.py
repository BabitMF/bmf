import bmf
import bmf.hmp as mp
from bmf import Log, LogLevel, Timestamp, VideoFrame
import torch
# Define custom Python modules for data conversion

class TensorToVideoFrame(bmf.Module):
    """
    Converts a Packet containing an image tensor (NHWC) to a Packet containing a VideoFrame.
    Supported sources:
    - hmp.Tensor carried by Packet
    - torch.Tensor carried by Packet
    """
    def process(self, task):
        input_queue = task.get_inputs()[0]
        output_queue = task.get_outputs()[0]

        while not input_queue.empty():
            pkt = input_queue.get()
            if pkt.timestamp == Timestamp.EOF:
                output_queue.put(bmf.Packet.generate_eof_packet())
                task.set_timestamp(Timestamp.DONE)
                return bmf.ProcessResult.OK

            # Pass-through if already a VideoFrame
            try:
                vf = pkt.get(VideoFrame)
                if vf is not None:
                    output_queue.put(pkt)
                    continue
            except Exception:
                pass

            # Extract an image tensor as torch.Tensor (NHWC)
            torch_tensor = None

            # Preferred: extract hmp.Tensor then convert to numpy directly
            try:
                hmp_tensor = pkt.get(mp.Tensor)
                if hmp_tensor is not None:
                    # Move to CPU and convert to numpy
                    np_img = hmp_tensor.cpu().numpy()
                    torch_tensor = torch.from_numpy(np_img)
            except Exception:
                pass

            # Fallback: direct python object may be hmp.Tensor or torch.Tensor
            if torch_tensor is None:
                try:
                    obj = pkt.get(None)
                    # Unwrap ("PYOBJ", x)
                    if isinstance(obj, tuple) and len(obj) >= 2 and obj[0] == 'PYOBJ':
                        obj = obj[1]
                    if isinstance(obj, mp.Tensor):
                        np_img = obj.cpu().numpy()
                        torch_tensor = torch.from_numpy(np_img)
                    elif isinstance(obj, torch.Tensor):
                        torch_tensor = obj
                except Exception:
                    pass

            if torch_tensor is None:
                # Unknown payload, forward as-is
                output_queue.put(pkt)
                continue

            # Ensure NHWC, uint8 [0,255]
            t = torch_tensor
            if t.dim() == 4:
                t = t[0]
            # If tensor seems NCHW (C first), move to NHWC based on channel dim heuristic
            if t.dim() == 3 and t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
                t = t.permute(1, 2, 0)
            # Move to CPU and uint8 [0,255]
            if t.dtype != torch.uint8:
                t = (t.float().clamp(0, 1) * 255.0).to(torch.uint8)
            if t.device.type != 'cpu':
                t = t.cpu()
            t = t.contiguous()

            # Build VideoFrame via hmp.Frame from numpy
            c = t.shape[-1] if t.dim() == 3 else 3
            pix_info = mp.PixelInfo(mp.kPF_RGB24 if c == 3 else (mp.kPF_RGBA if c == 4 else mp.kPF_GRAY8))

            try:
                np_img = t.numpy()
                hmp_tensor = mp.from_numpy(np_img)
                frame = mp.Frame(hmp_tensor, pix_info)
                vf = VideoFrame(frame)
                out_pkt = bmf.Packet(vf)
                out_pkt.timestamp = pkt.timestamp
                output_queue.put(out_pkt)
            except Exception as e:
                Log.log(LogLevel.ERROR, f"TensorToVideoFrame failed to create VideoFrame: {e}")
                output_queue.put(bmf.Packet.generate_eof_packet())
                task.set_timestamp(Timestamp.DONE)
                return bmf.ProcessResult.ERROR

        return bmf.ProcessResult.OK


class VideoFrameToTensor(bmf.Module):
    """
    Converts a Packet containing a VideoFrame back to an NHWC torch.Tensor in [0,1], batched.
    """
    def process(self, task):
        input_queue = task.get_inputs()[0]
        output_queue = task.get_outputs()[0]

        while not input_queue.empty():
            pkt = input_queue.get()
            if pkt.timestamp == Timestamp.EOF:
                output_queue.put(bmf.Packet.generate_eof_packet())
                task.set_timestamp(Timestamp.DONE)
                return bmf.ProcessResult.OK

            try:
                vf = pkt.get(VideoFrame)
                if vf is None:
                    # Not a VideoFrame - forward as-is
                    output_queue.put(pkt)
                    continue
                # Convert VideoFrame to numpy in RGB24, then to torch
                rgb = mp.PixelInfo(mp.kPF_RGB24)
                np_img = vf.reformat(rgb).frame().plane(0).numpy()
                torch_img = torch.from_numpy(np_img)
                if torch_img.dtype != torch.float32:
                    torch_img = torch_img.float().div_(255.0)
                if torch_img.dim() == 3:
                    torch_img = torch_img.unsqueeze(0)
                out_pkt = bmf.Packet(torch_img)
                out_pkt.timestamp = pkt.timestamp
                output_queue.put(out_pkt)
            except Exception as e:
                Log.log(LogLevel.ERROR, f"VideoFrameToTensor failed: {e}")
                output_queue.put(bmf.Packet.generate_eof_packet())
                task.set_timestamp(Timestamp.DONE)
                return bmf.ProcessResult.ERROR

        return bmf.ProcessResult.OK
