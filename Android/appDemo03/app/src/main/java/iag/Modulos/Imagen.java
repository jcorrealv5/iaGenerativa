package iag.Modulos;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Base64;
import android.widget.ImageView;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
public class Imagen {
    public static void mostrarDesdeBase64(String base64, ImageView iv)
    {
        byte[] decodedString = Base64.decode(base64, Base64.DEFAULT);
        Bitmap bmp = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
        iv.setImageBitmap(bmp);
    }

    public static void mostrarDesdeBytes(byte[] buffer, ImageView iv)
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        InputStream is = new ByteArrayInputStream(buffer);
        Bitmap bmp = BitmapFactory.decodeStream(is);
        iv.setImageBitmap(bmp);
    }
}
