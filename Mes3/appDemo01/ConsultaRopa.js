var idAnimacion;

window.onload = function () {
    btnGenerar.onclick = function () {
        idAnimacion = null;
        generarRopa();
    }
    btnGenerarAnimado.onclick = function () {
        idAnimacion = requestAnimationFrame(generarRopa);
    }
}

async function generarRopa() {
    var rptaHttp = await fetch("GenerarRopa", { method: "GET" });
    if (rptaHttp.ok) {
        var rptaBlob = await rptaHttp.blob();
        if (rptaBlob.size > 0) {
            imgRopa.src = URL.createObjectURL(rptaBlob);
            if (idAnimacion != null) idAnimacion = requestAnimationFrame(generarRopa);
        }
    }
}