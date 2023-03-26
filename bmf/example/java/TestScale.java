import com.bytedance.hmp.ChannelFormat;
import com.bytedance.hmp.PixelInfo;
import com.google.gson.*;
import com.bytedance.bmf.*;


public class TestScale
{
    public static void main(String[] args) throws Exception
    {

        JsonObject option = new JsonObject();
        option.addProperty("input_path", "../files/img.mp4");
        ModuleInfo moduleInfo = new ModuleInfo("c_ffmpeg_decoder", "", "", "");
        Class itypes[] = new Class[]{};
        Class otypes[] = new Class[]{VideoFrame.class};
        ModuleFunctor decoder = new ModuleFunctor(moduleInfo, option, itypes, otypes);

        String optStr = "{\"filters\":[{\"inputs\":[{\"pin\":0,\"stream\":0}],\"name\":\"scale\",\"outputs\":[{\"pin\":0,\"stream\":0}],\"para\":\"100:200\"}]}";
        Gson gson = new Gson();
        JsonObject option4 = gson.fromJson(optStr, JsonObject.class);
        ModuleInfo moduleInfo4 = new ModuleInfo("c_ffmpeg_filter", "", "", "");
        Class itypes4[] = new Class[]{VideoFrame.class};
        Class otypes4[] = new Class[]{VideoFrame.class};
        ModuleFunctor filter = new ModuleFunctor(moduleInfo4, option4, itypes4, otypes4);

        JsonObject option3 = new JsonObject();
        option3.addProperty("output_path", "output.mp4");
        ModuleInfo moduleInfo3 = new ModuleInfo("c_ffmpeg_encoder", "", "", "");
        Class itypes3[] = new Class[]{VideoFrame.class};
        Class otypes3[] = new Class[]{};
        ModuleFunctor encoder = new ModuleFunctor(moduleInfo3, option3, itypes3, otypes3);

        while (true){
            VideoFrame[] vfs = decode(decoder);
            if (vfs == null){
                decoder.free();
                filter.free();
                encoder.free();
                break;
            }
            for (VideoFrame vf : vfs){
                try {
                    vf = (VideoFrame)filter.call(vf)[0];
                    System.out.printf("videoframe pts is : %d\n", vf.pts());
                    encoder.call(vf);
                } catch (ProcessDoneException e) {
                    e.printStackTrace();
                    break;
                }
            }
        }
    }

    public static VideoFrame[] decode(ModuleFunctor decoder) throws Exception
    {
        try {
            decoder.execute(null);
            Object[] outputs = decoder.fetch(0);
            VideoFrame[] vfs = new VideoFrame[outputs.length];
            for (int i=0; i<outputs.length; i++){
                vfs[i] = (VideoFrame)outputs[i];
            }
            return vfs;
        } catch (ProcessDoneException e){
            return null;
        } catch (Exception e) {
            throw new Exception("decode failed");
        }
    }
}