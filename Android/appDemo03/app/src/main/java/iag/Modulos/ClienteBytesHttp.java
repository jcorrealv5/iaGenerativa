package iag.Modulos;
import iag.Interfaces.CallbackAsync;
import android.util.Log;
import java.util.concurrent.ExecutionException;

public class ClienteBytesHttp {
    public static byte[] get(String url,CallbackAsync cba)
    {
        byte[] rpta = null;
        try {
            rpta = new ConexionBytesHttp(cba).execute(url).get();
        }
        catch (ExecutionException ex) {
            rpta=("Error WS: " + ex.getMessage()).getBytes();
            Log.e("logHttp", ex.getMessage());
        }
        catch (InterruptedException ex)
        {
            rpta= ("Error WS: " + ex.getMessage()).getBytes();
            Log.e("logHttp", ex.getMessage());
        }
        return rpta;
    }

    public static byte[] post(String url,CallbackAsync cba, byte[] data)
    {
        byte[] rpta = null;
        try {
            rpta = new ConexionBytesHttp(cba, data).execute(url).get();
        }
        catch (ExecutionException ex) {
            rpta = ("Error WS: " + ex.getMessage()).getBytes();
            Log.e("logHttp", ex.getMessage());
        }
        catch (InterruptedException ex)
        {
            rpta = ("Error WS: " + ex.getMessage()).getBytes();
            Log.e("logHttp", ex.getMessage());
        }
        return rpta;
    }
}
