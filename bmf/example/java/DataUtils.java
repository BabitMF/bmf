import com.bytedance.hmp.Ptr;
import com.bytedance.hmp.ScalarType;
import com.bytedance.hmp.ChannelFormat;
import com.bytedance.hmp.Image;
import com.bytedance.hmp.Frame;
import com.bytedance.hmp.Tensor;
import com.bytedance.hmp.PixelFormatDesc;
import com.bytedance.hmp.PixelInfo;
import com.bytedance.hmp.PixelFormat;
import com.bytedance.hmp.DeviceType;
import com.google.gson.*;
import com.bytedance.bmf.*;


public class DataUtils
{
    public static Packet load_from_file(String fileName, String type, JsonObject jsonObj) throws Exception
    {
        if (type.equals("image") || type.equals("frame")) {
            ScalarType dtype = ScalarType.kUInt8;
            int width = jsonObj.get("width").getAsInt();
            int height = jsonObj.get("height").getAsInt();
            String ext = fileName.substring(fileName.lastIndexOf("."));
            Tensor data = Tensor.fromFile(fileName, dtype, -1, 0);
            if (ext.equals(".rgb")){
                if (jsonObj.get("format").getAsString().equals("rgba")) {
                    data = data.reshape(new long[]{height, width, 4});
                    VideoFrame vf = new VideoFrame(new Image(data, ChannelFormat.kNHWC));
                    return new Packet(vf);
                } else {
                    data = data.reshape(new long[]{height, width, 3});
                    if (type.equals("image")){
                        VideoFrame vf = new VideoFrame(new Image(data, ChannelFormat.kNHWC));
                        return new Packet(vf);
                    } else {
                        Tensor[] planes = new Tensor[]{data.select(2, 0), data.select(2, 1), data.select(2, 2)};
                        PixelInfo pix_info = new PixelInfo(PixelFormat.PF_RGB24);
                        VideoFrame vf = new VideoFrame(new Frame(planes, pix_info));
                        return new Packet(vf);
                    }
                }
            } else if (ext.equals(".yuv")) {
                String format_str = jsonObj.get("format").getAsString();
                PixelFormat pformat;
                if(format_str.equals("yuv420p")){
                    pformat = PixelFormat.PF_YUV420P;
                } else if (format_str.equals("yuva420p")){
                    pformat = PixelFormat.PF_YUVA420P;
                } else {
                    throw new Exception(String.format("Unsupport image/frame format %s", format_str));
                }

                PixelFormatDesc desc = new PixelFormatDesc(pformat);
                if (desc.infer_nitems(width, height) != data.nitems()) {
                    throw new Exception("Invalid image size");
                }
                Tensor[] planes = new Tensor[desc.nplanes()];
                int off = 0;
                for(int i=0; i<desc.nplanes(); i++) {
                    int n = desc.infer_nitems(width, height, i);
                    int w = desc.infer_width(width, i);
                    int h = desc.infer_height(height, i);
                    Tensor plane = data.slice(0, off, off + n, 1).reshape(new long[]{h, w, -1});
                    planes[i] = plane;
                    off += n;
                }

                VideoFrame vf = new VideoFrame(new Frame(planes, width, height, new PixelInfo(pformat)));
                return new Packet(vf);
            } else {
                throw new Exception(String.format("Unsupport image/frame file format %s", ext));
            }
        } else {
            throw new Exception(String.format("Unsupported data type %s", type));
        }
    }

    public static void store_to_file(Packet packet, String fileName) throws Exception
    {
        if (packet.is(VideoFrame.class)){
            VideoFrame vf = (VideoFrame)packet.get(VideoFrame.class);
            Tensor data;
            if (vf.isImage()){
                data = vf.image().data();
            } else {
                Frame frame = vf.frame();
                int nitems = 0;
                for (int i=0; i<frame.nplanes(); i++){
                    nitems += frame.plane(i).nitems();
                }

                //concatenate all planes
                data = Tensor.empty(new long[]{nitems}, frame.plane(0).dtype(), "cpu", false);
                nitems = 0;
                for (int i=0; i<frame.nplanes(); i++){
                    Tensor plane = frame.plane(i);
                    Tensor sTensor = data.slice(0, nitems, nitems + plane.nitems(), 1);
                    sTensor.copyFrom(plane.reshape(new long[]{-1}));
                    sTensor.free();
                    nitems += plane.nitems();
                }
            }
            data.toFile(fileName);
        } else {
            throw new Exception("Unsupported data type");
        }
    }
}