package aig.appdemo01;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;
import aig.Interfaces.CallbackAsync;
import aig.Modulos.ClienteBytesHttp;
import aig.Modulos.Imagen;

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
                obtenerRopaServicio(false);
            }
        });
        Button btnGenerarAnimado = findViewById(R.id.btnGenerarAnimado);
        btnGenerarAnimado.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(btnGenerarAnimado.getText().equals("Generar Animado")){
                    btnGenerarAnimado.setText("Detener Animado");
                    detener=false;
                    obtenerRopaServicio(true);
                }
                else{
                    btnGenerarAnimado.setText("Generar Animado");
                    detener=true;
                }
            }
        });
    }

    private void obtenerRopaServicio(boolean recursivo){
        String urlServicio = getResources().getString(R.string.urlServicioWeb);
        String urlMetodo = urlServicio + "GenerarRopa";
        ClienteBytesHttp.get(urlMetodo, new CallbackAsync() {
            @Override
            public void MostrarRespuestaBytesAsync(byte[] rpta) {
                if(rpta!=null && rpta.length>0){
                    ImageView imgFoto =findViewById(R.id.imgRopa);
                    Imagen.mostrarDesdeBytes(rpta, imgFoto);
                    if(recursivo && !detener) obtenerRopaServicio(true);
                }
            }
        });
    }
}