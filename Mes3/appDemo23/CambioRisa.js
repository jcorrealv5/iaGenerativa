window.onload = function () {
	var ctx = canvas.getContext("2d");
	var canvasAncho = canvas.width;
	var canvasAlto = canvas.height;
	ctx.fillStyle = "black";
	ctx.fillRect(0, 0, canvasAncho, canvasAlto);

	var listaRisa = ["|Seleccione", "0|Sin Risa", "1|Con Risa"];
	crearCombo(listaRisa, cboRisa);

	btnActivarCamara.onclick = function () {
		if (btnActivarCamara.value == "Activar Camara") {
			btnActivarCamara.value = "Pausar Video";
			btnTomarFoto.style.display = "inline";
			activarCamara();
		}
		else {
			btnActivarCamara.value = "Activar Camara";
			btnTomarFoto.style.display = "none";
			video.pause();
		}
	}

	btnTomarFoto.onclick = function () {
		ctx.drawImage(video, 0, 0, canvasAncho, canvasAlto);
	}
	
	btnCambiarRisa.onclick = async function () {
		if (cboRisa.value != "") {
			var risa = cboRisa.value;
			var imgBase64 = canvas.toDataURL().replace("data:image/png;base64,", "");
			var blob = convertBase64ToBlob(imgBase64);
			var token = document.getElementsByName("csrfmiddlewaretoken")[0].value;
			var frm = new FormData();
			frm.append("csrfmiddlewaretoken", token);
			frm.append("risa", risa);
			frm.append("archivo", blob);				
			var rptaHttp = await fetch("CambiarRisa", { method: "POST", body: frm });
			if (rptaHttp.ok) {
				var blob = await rptaHttp.blob();
				var nTotal = blob.size;
				if (nTotal > 0) {
					var blobImagen = new Blob([blob], { "type": "image/jpg" });
					imgRisaCambiada.src = URL.createObjectURL(blobImagen);
				}
			}
		}
		else alert("Selecciona la Risa");
	}

	btnNuevo.onclick = function () {
		cboRisa.value = "";
		ctx.fillStyle = "black";
		ctx.fillRect(0, 0, canvasAncho, canvasAlto);
		imgRisaCambiada.src = "";
    }
}

async function activarCamara() {
	try {
		const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
		video.srcObject = stream;
		video.play();
	}
	catch (error) {
		console.log('Error:', error);
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

function convertBase64ToBlob(imagenBase64) {
	var blob = null;
	var bytesCaracteres = atob(imagenBase64);
	var buffer = new Array(bytesCaracteres.length);
	for (let i = 0; i < bytesCaracteres.length; i++) {
		buffer[i] = bytesCaracteres.charCodeAt(i);
	}
	var byteArray = new Uint8Array(buffer);
	blob = new Blob([byteArray], { type: "image/jpeg" });
	return blob;
}