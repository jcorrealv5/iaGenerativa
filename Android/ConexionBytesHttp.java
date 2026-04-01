package cursoAndroid.Modulos;

import cursoAndroid.Interfaces.CallbackAsync;
import android.os.AsyncTask;
import android.util.Log;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class ConexionBytesHttp  extends AsyncTask<String, Void, byte[]> {
    byte[] Data = null;
    CallbackAsync cba;

    public ConexionBytesHttp(CallbackAsync cba)
    {
        this.cba = cba;
    }

    public ConexionBytesHttp(CallbackAsync cba, byte[] data)
    {
        this.Data = data;
        this.cba = cba;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
    }

    @Override
    protected byte[] doInBackground(String...urls) {
        byte[] rpta = null;
        try {
            boolean esPost = (this.Data!=null);
            URL url = new URL(urls[0]);
            HttpURLConnection urlConnection = (HttpURLConnection) url.openConnection();
            urlConnection.setConnectTimeout(15000);
            if(esPost) {
                urlConnection.setRequestMethod("POST");
                urlConnection.setDoInput(true);
                urlConnection.setDoOutput(true);
                urlConnection.setFixedLengthStreamingMode(this.Data.length);
                urlConnection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
                OutputStream outputStream = urlConnection.getOutputStream();
                outputStream.write(this.Data);
                outputStream.flush();
                outputStream.close();
                urlConnection.connect();
            }
            InputStream is = urlConnection.getInputStream();
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[1024];
            while ((nRead = is.read(data, 0, data.length)) != -1) {
                baos.write(data, 0, nRead);
            }
            rpta = baos.toByteArray();
        }
        catch (Exception ex)
        {
            String rptaString = "Error - " + ex.getMessage();
            rpta = rptaString.getBytes();
            Log.e("logHttp", ex.getMessage());
        }
        return rpta;
    }

    protected void onPostExecute(byte[] result) {
        super.onPostExecute(result);
        this.cba.MostrarRespuestaBytesAsync(result);
    }
}