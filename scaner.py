from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time

URL_LOGIN = "https://seguroyfacil.com/inicia-sesion"
# DOCUMENTO = "123456789"
PASSWORD = "tu_password"

# lista de cédulas
cedulas = [
    "12345678",
    "12345679",
    "12345680",
    "12345681",
    "12345682",
    "12345683",
    "12345684",
    "12345685",
    "12345686",
    "12345687",
    "12345688",
    "12345689",
    "12345690",
    "12345691",
    "12345692",
    "12345693",
    "12345694",
    "12345695",
    "12345696",
    "12345697",
    "12345698",
    "12345699",
    "12345700",
    "12345701",
    "12345702",
    "12345703",
    "12345704",
    "12345705",
    "12345706",
    "12345707",
    "12345708",
    "12345709",
    "12345710",
    "12345711",
    "12345712",
    "12345713",
    "12345714",
    "12345715",
    "12345716",
    "12345717",
    "12345718",
    "12345719",
    "12345720",
    "12345721",
    "12345722",
    "12345723",
    "12345724",
    "12345725",
    "12345726",
    "12345727",
    "12345728",
    "12345729"
]

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.maximize_window()

wait = WebDriverWait(driver, 20)

for cedula in cedulas:

    print(f"Procesando cédula: {cedula}")

    driver.get(URL_LOGIN)

    time.sleep(3)

    # INPUT DOCUMENTO
    input_documento = wait.until(
        EC.visibility_of_element_located(
            (By.XPATH, "//input[@placeholder='Ingresa tu número de documento']")
        )
    )

    input_documento.clear()
    input_documento.send_keys(cedula)

    # INPUT PASSWORD
    input_password = wait.until(
        EC.visibility_of_element_located(
            (By.XPATH, "//input[@placeholder='Ingresa tu contraseña']")
        )
    )

    input_password.clear()
    input_password.send_keys(PASSWORD)

    time.sleep(2)

    # BOTON LOGIN
    boton_login = wait.until(
        EC.element_to_be_clickable(
            (By.XPATH, "//button[.//div[contains(text(),'INICIAR SESIÓN')]]")
        )
    )

    boton_login.click()

    print(f"Login realizado con {cedula}")

    # esperar resultado
    time.sleep(5)

print("Proceso terminado")

driver.quit()