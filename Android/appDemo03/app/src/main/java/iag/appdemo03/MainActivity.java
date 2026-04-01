package iag.appdemo03;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;
import iag.Interfaces.CallbackAsync;
import iag.Modulos.ClienteBytesHttp;
import iag.Modulos.Imagen;

public class MainActivity extends AppCompatActivity {
    boolean detener;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button btnGenerar = findViewById(R.id.btnGenerar);
        btnGenerar.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                servicioGenerarRostro(false);
            }
        });
        Button btnGenerarAnimado = findViewById(R.id.btnGenerarAnimado);
        btnGenerarAnimado.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(btnGenerarAnimado.getText().equals("Generar Animado")){
                    btnGenerarAnimado.setText("Detener Animado");
                    detener=false;
                    servicioGenerarRostro(true);
                }
                else{
                    btnGenerarAnimado.setText("Generar Animado");
                    detener=true;
                }
            }
        });
    }

    private void servicioGenerarRostro(boolean recursivo){
        String urlServicio = getResources().getString(R.string.urlServicioWeb);
        String urlMetodo = urlServicio + "GenerarRostro";
        ClienteBytesHttp.get(urlMetodo, new CallbackAsync() {
            @Override
            public void MostrarRespuestaBytesAsync(byte[] rpta) {
                if(rpta!=null && rpta.length>0){
                    ImageView imgCara =findViewById(R.id.imgCara);
                    Imagen.mostrarDesdeBytes(rpta, imgCara);
                    if(recursivo && !detener) servicioGenerarRostro(true);
                }
            }
        });
    }
}