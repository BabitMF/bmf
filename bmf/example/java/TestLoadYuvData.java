import com.bytedance.hmp.ChannelFormat;
import com.bytedance.hmp.PixelInfo;
import com.google.gson.*;
import com.bytedance.bmf.*;
import java.io.File;
import java.util.Arrays;


public class TestLoadYuvData
{
    public static void main(String[] args) throws Exception
    {
        String src_video = "../files/img.mp4";
        String input_path = "yuv_data";
        String output_path = "yuv_out";

        File input_dir = new File(input_path);
        if (!input_dir.exists()){
            input_dir.mkdirs();
            video_to_yuv(src_video, input_path);
        }

        File file = new File(input_path);
        File[] tempList = file.listFiles();
        Arrays.sort(tempList);

        File dir = new File(output_path);
        if (!dir.exists()){
            dir.mkdirs();
        }

        JsonObject param = new JsonObject();
        param.addProperty("format", "yuv420p");
        param.addProperty("width", 1920);
        param.addProperty("height", 1080);
        for (File f : tempList){
            String fn = f.toString();
            VideoFrame vf = load_data(fn, param);
            System.out.printf("%s , frm size is : %d x %d\n", fn, vf.width(), vf.height());
            String suffix = fn.substring(fn.lastIndexOf("/"));
            Packet pkt = new Packet(vf);

            // save yuv data to file
            DataUtils.store_to_file(pkt, output_path + suffix);
        }
    }

    public static VideoFrame load_data(String file_name, JsonObject param)
    {
        // load yuv data from file
        Packet pkt_frm;
        try {
            pkt_frm = DataUtils.load_from_file(file_name, "frame", param);
            VideoFrame vf_frm = (VideoFrame)pkt_frm.get(VideoFrame.class);
            return vf_frm;
        } catch (Exception e){
            e.printStackTrace();
            return null;
        }
    }

    public static void video_to_yuv(String src_video, String input_path) throws Exception
    {
        File input_dir = new File(input_path);

        String FFMPEG="ffmpeg  -hide_banner -loglevel error";
        String cmd = FFMPEG + " -i " + src_video + " -vsync 0 -frame_pts true " + input_path + "/out-%04d.png";
        System.out.println(src_video + " to png");
        Runtime.getRuntime().exec(cmd).waitFor();

        File[] tempList = input_dir.listFiles();
        for (File f: tempList){
            String fn = f.toString();
            String prefix = fn.substring(0, fn.lastIndexOf("."));
            System.out.println(prefix + ".png to yuv");
            cmd = FFMPEG + " -i " + fn + " -pix_fmt yuv420p " + prefix + ".yuv";
            Runtime.getRuntime().exec(cmd).waitFor();
            cmd = "rm " + fn;
            Runtime.getRuntime().exec(cmd).waitFor();
        }
    }
}