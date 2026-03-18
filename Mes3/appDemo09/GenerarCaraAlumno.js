var blobs = [];
var nBlob = 0;
var cBlob = 0;
var idAnima;

window.onload = function () {
	var listaAlumnos = ["0|Carlos", "1|Edelson", "2|Juan Carlos", "3|Luis", "4|Pedro"];
	crearCombo(listaAlumnos, cboCaraInicio);
	crearCombo(listaAlumnos, cboCaraFin);
	cboCaraInicio.value = "0";
	cboCaraFin.value = "1";

	btnGenerar.onclick = function () {
		cBlob = 0;
		blobs = [];
        generarCaras();
	}

	btnNuevo.onclick = function () {
		cBlob = 0;
		blobs = [];
		cboCaraInicio.value = "0";
		cboCaraFin.value = "1";
		txtNumMuestras.value = "10";
		imgCara.src = "";
		spnProgreso.innerText = "";
    }
}

async function generarCaras() {
	var caraInicio = cboCaraInicio.value;
	var caraFin = cboCaraFin.value;
	var nMuestras = txtNumMuestras.value;
	var url = "GenerarCaras?caraInicio=" + caraInicio + "&caraFin=" + caraFin + "&nMuestras=" + nMuestras;
	var rptaHttp = await fetch(url, { method: "GET" });
	if (rptaHttp.ok) {
        var blob = await rptaHttp.blob();
		var nTotal = blob.size;
		if (nTotal > 0) {
			//Obtener el Byte 1
			var byte1 = blob.slice(0, 1);
			var buffer1 = await byte1.arrayBuffer();
			var array1 = new Uint8Array(buffer1);
			var n1 = array1[0];
			//Obtener el Byte 2
			var byte2 = blob.slice(1, 2);
			var buffer2 = await byte2.arrayBuffer();
			var array2 = new Uint8Array(buffer2);
			var n2 = array2[0];
			//Obtener el Tamanio del Texto
			var nTexto = (n1 * 255) + n2
			//Obtener el Texto
			var bytesTexto = blob.slice(2, 2 + nTexto);
			var bufferTexto = await bytesTexto.arrayBuffer();
			var arrayTexto = new Uint8Array(bufferTexto);
			var texto = "";
			for (var i = 0; i < nTexto; i++) {
				texto += String.fromCharCode(arrayTexto[i]);
			}
			var lista = texto.split("|");
			var nRegistros = lista.length;
			var x = 2 + nTexto;
			var size, bytesImagen;
			nBlob = nRegistros;
			cBlob = 0;
			for (var i = 0; i < nRegistros; i++) {
				size = +lista[i];
				bytesImagen = blob.slice(x, x + size);
				var blobImagen = new Blob([bytesImagen], { "type": "image/jpg" });
				blobs.push(blobImagen);
				x += size;
			}
			idAnima = setInterval(mostrarCara, 100);
		}
    }
}

function crearCombo(lista, cbo, primerItem) {
	var primerItem = (primerItem == null ? "" : primerItem);
	var html = "";
	var nRegistros = lista.length;
	var campos = [];
	if (primerItem != "") {
		html += "<option value=''>";
		html += primerItem;
		html += "</option>";
	}
	for (var i = 0; i < nRegistros; i++) {
		campos = lista[i].split("|");
		html += "<option value='";
		html += campos[0];
		html += "'>";
		html += campos[1];
		html += "</option>";
	}
	cbo.innerHTML = html;
}

function mostrarCara() {
	spnProgreso.innerText = (cBlob + 1) + " de " + nBlob;
	imgCara.src = URL.createObjectURL(blobs[cBlob]);
	cBlob++;
	if (cBlob == nBlob) clearInterval(idAnima);
}