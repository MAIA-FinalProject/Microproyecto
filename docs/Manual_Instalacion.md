# Manual de Instalación y Reproducibilidad — NeuroRisk

## 1. Alcance y Arquitectura

Este manual describe el procedimiento detallado para reproducir y desplegar la solución **NeuroRisk** en un entorno local, integrando el almacenamiento y versionado de datos en Google Drive (vía DVC) y el servidor de seguimiento de modelos (MLflow Tracking Server) alojado en la nube en AWS EC2.

La arquitectura de despliegue se compone de dos capas:

- **Capa de Tracking (AWS EC2)**:
  - Aloja el **MLflow Tracking Server** conectado a una base de datos SQLite y almacén de artefactos local en la instancia.
  - Se ejecuta de manera continua y persistente mediante un servicio `systemd`, iniciando automáticamente tras cada reinicio o encendido de la máquina.
  - Expone la interfaz gráfica y la API REST de MLflow a través de una dirección IP Elástica en el puerto `5000`.

- **Capa de Servicio y Aplicación (Entorno Local — Docker Compose)**:
  - **Backend (FastAPI)**: Microservicio REST que descarga el artefacto del modelo entrenado (`.joblib`) directamente desde el servidor de MLflow en EC2 al iniciar, exponiendo los endpoints `/health` y `/predict` en el puerto `8000`.
  - **Frontend (Streamlit)**: Tablero interactivo para el personal de seguimiento clínico en el puerto `8501`, el cual permite ingresar parámetros de neonatos y madres para obtener el riesgo estimado en tiempo real.

---

## 2. Requisitos Previos

Para desplegar y ejecutar el proyecto localmente se requiere:

- **Git** instalado en el sistema.
- **Docker** y **Docker Compose** (Docker Desktop en Windows/macOS o Docker Engine en Linux).
- **DVC (Data Version Control)** para la descarga de datos versionados.
- **Acceso a Google Drive**: Permisos de lectura en la carpeta compartida del proyecto y el `client_secret` de Google Drive proporcionado por el equipo.
- **Conectividad con AWS EC2**: Acceso a la IP Elástica del servidor MLflow (`44.215.228.113`).
- *(Opcional)* **Python 3.12** y el gestor **`uv`** (si se desea ejecutar, probar o desarrollar fuera de contenedores Docker).

---

## 3. Servidor de MLflow en AWS EC2

### 3.1 Especificaciones de la Instancia
- **Proveedor / Región**: AWS / `us-east-1` (N. Virginia).
- **Sistema Operativo**: Amazon Linux 2023.
- **Tipo de Instancia**: `t3.small` (2 vCPUs, 2 GB RAM).
- **IP Elástica Asignada**: `44.215.228.113`.
- **URL del Tracking Server**: `http://44.215.228.113:5000`.

### 3.2 Reglas del Security Group (Firewall)
El Security Group asignado a la instancia debe tener habilitadas las siguientes reglas de entrada (*Inbound Rules*):

| Tipo | Protocolo | Puerto | Origen (*Source*) | Propósito |
|---|---|---|---|---|
| **Custom TCP** | TCP | `5000` | `0.0.0.0/0` | Acceso a la interfaz web y API de MLflow |
| **SSH** | TCP | `22` | `0.0.0.0/0` o IP de administración | Administración SSH y EC2 Instance Connect |
| **Custom TCP** | TCP | `8000` | `0.0.0.0/0` | *(Preparado)* API FastAPI si se hospeda en EC2 |
| **Custom TCP** | TCP | `8501` | `0.0.0.0/0` | *(Preparado)* Dashboard Streamlit si se hospeda en EC2 |

### 3.3 Servicio Persistente `systemd` y Configuración CORS
El servidor MLflow corre gestionado por `systemd` bajo el archivo `/etc/systemd/system/mlflow.service`. Cuenta con la variable de entorno `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS=*` configurada para permitir que la interfaz web opere sin bloqueos de origen cruzado (CORS 403) al accederse vía IP pública:

```ini
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Environment="MLFLOW_SERVER_CORS_ALLOWED_ORIGINS=*"
User=ec2-user
Group=ec2-user
WorkingDirectory=/home/ec2-user/mlflow
Environment="PATH=/home/ec2-user/mlflow-venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ec2-user/mlflow-venv/bin/mlflow server --host 0.0.0.0 --port 5000 --workers 1 --backend-store-uri sqlite:////home/ec2-user/mlflow/mlflow.db --default-artifact-root /home/ec2-user/mlflow/artifacts --allowed-hosts "44.215.228.113:5000"
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

> **Nota de Persistencia**: Gracias a `systemd`, cada vez que la instancia EC2 se encienda o se reinicie, el servidor MLflow arrancará de forma automática sin necesidad de intervención manual por SSH.

### 3.4 Verificación del Servidor
Abra en el navegador:
```text
http://44.215.228.113:5000
```
Verifique que cargue la lista de experimentos y que el experimento `preterm_infant_prediction` (ID `1`) esté disponible.

---

## 4. Clonar el Repositorio

Abra una terminal en su equipo local y clone el repositorio:

```bash
git clone https://github.com/MAIA-FinalProject/Microproyecto.git
cd Microproyecto
```

---

## 5. Configurar Variables de Entorno (`.env`)

Copie la plantilla base:

En Linux, macOS o Git Bash:
```bash
cp .env.example .env
```

En Windows PowerShell:
```powershell
Copy-Item .env.example .env
```

Abra el archivo `.env` y configure las variables de conexión:

```dotenv
ENVIRONMENT=development
LOG_LEVEL=INFO

# Servidor MLflow en AWS EC2
MLFLOW_TRACKING_URI=http://44.215.228.113:5000
MLFLOW_EXPERIMENT_NAME=preterm_infant_prediction

# Modelo a servir (Logistic Regression baseline probado)
MODEL_URI=runs:/2a7ea164ec7f425692eca3d9c5347f1b/model

# Configuración de red local
API_HOST=0.0.0.0
API_PORT=8000
API_URL=http://localhost:8000
```

### Modelos Disponibles en el Servidor MLflow
En el Tracking Server de EC2 se encuentran registrados los siguientes modelos entrenados listos para servir:

| Algoritmo | Run ID | `MODEL_URI` |
|---|---|---|
| **Logistic Regression** (Default) | `2a7ea164ec7f425692eca3d9c5347f1b` | `runs:/2a7ea164ec7f425692eca3d9c5347f1b/model` |
| **XGBoost** | `d408718ae6e345e6b1464ea91e71485c` | `runs:/d408718ae6e345e6b1464ea91e71485c/model` |
| **Random Forest** | `b899650c2e8a4837835477539f03ece8` | `runs:/b899650c2e8a4837835477539f03ece8/model` |

Para cambiar el modelo servido por la API, modifique la variable `MODEL_URI` en el archivo `.env` y reinicie los contenedores.

---

## 6. Configurar DVC y Descargar los Datos

El conjunto de datos crudo (`data/raw/`) se encuentra versionado con DVC y respaldado en Google Drive.

1. Configure de forma local (una única vez) el client secret compartido por el equipo:
   ```bash
   dvc remote modify --local gdrive_remote gdrive_client_secret "<CLIENT_SECRET_COMPARTIDO>"
   ```
   *(Este comando almacena la clave en `.dvc/config.local`, archivo ignorado por git por seguridad).*

2. Descargue los datos desde el almacenamiento remoto:
   ```bash
   dvc pull
   ```
   *Durante la primera ejecución se abrirá una ventana de navegador para autorizar la lectura con una cuenta de Google con acceso a la carpeta compartida.*

3. Confirme que los archivos se hayan descargado en la carpeta `data/raw/`:
   - `Dataset.xlsx`
   - `Dataset.sav`
   - `Codebook.pdf`

---

## 7. Despliegue con Docker Compose

Desde la raíz del repositorio, construya y levante los contenedores:

```bash
docker compose -f deploy/docker-compose.yml up --build -d
```

Este comando orquesta dos servicios en una red interna compartida (`ml_network`):
- **`api`**: Contenedor FastAPI que expone el puerto `8000`.
- **`dashboard`**: Contenedor Streamlit que expone el puerto `8501`.

### Verificar Estado de los Contenedores
```bash
docker compose -f deploy/docker-compose.yml ps
```
Ambos servicios deben figurar con estado `Up` o `running`.

### Consultar Logs
En caso de requerir depuración:
```bash
docker compose -f deploy/docker-compose.yml logs -f api
docker compose -f deploy/docker-compose.yml logs -f dashboard
```

---

## 8. URLs de Acceso

| Servicio | URL Local | Descripción |
|---|---|---|
| **Dashboard Streamlit** | `http://localhost:8501` | Interfaz interactiva de soporte clínico |
| **Documentación API (Swagger)** | `http://localhost:8000/docs` | Explorador interactivo OpenAPI de la API |
| **Health Check API** | `http://localhost:8000/health` | Estado del backend y confirmación de carga del modelo |
| **MLflow Tracking Server** | `http://44.215.228.113:5000` | Servidor central de experimentos y artefactos en EC2 |

---

## 9. Validación del Despliegue y el Modelo

Abra en su navegador la URL del health check:
```text
http://localhost:8000/health
```

Cuando el backend se conecta con éxito a MLflow y carga el modelo `.joblib`, responderá:

```json
{
  "status": "ok",
  "model_loaded": true,
  "features_loaded": true,
  "expected_features": [
    "Sex", "DM", "preeclampsia", "hypothyroid", "PROM", "IUGR",
    "pregnancycomplication", "pneumothorax", "NEC", "sepsis", "PDA",
    "icter", "meningitis", "IVH", "seizure", "BPD", "apgar1", "apgar5",
    "B.C", "PregnancyAge", "drug.mother", "duration.hopitalization",
    "ehya.badve.tavallod", "BirthWeight", "RoundHeadAtBirth",
    "notaggressive.ventilation", "csf.culture", "congenital.anomaly",
    "duration.O2", "SepsisnegativeCulture", "mother.sonogarphy.result",
    "intervencion_respiratoria_agresiva", "laborType_NVD", "laborType_cs",
    "type.of.ressucitation_advanced", "type.of.ressucitation_no_resuscitation",
    "type.of.ressucitation_ppv"
  ],
  "detail": null
}
```

### Diagnóstico de Problemas (`status: degraded`)
Si la respuesta indica `"status": "degraded"` o `"model_loaded": false`, verifique:
1. Que la instancia EC2 esté en estado *Running*.
2. Que `http://44.215.228.113:5000` responda en el navegador.
3. Que `MODEL_URI` en `.env` contenga un Run ID válido y existente.
4. Si modificó `.env`, reinicie los contenedores:
   ```bash
   docker compose -f deploy/docker-compose.yml restart api
   ```

---

## 10. Operación del Tablero: Modo Mock vs. Modo Live

1. Abra el tablero en `http://localhost:8501`.
2. En la barra lateral izquierda (**Configuración**):
   - **Desactive** el interruptor **Modo Simulación (Mock API)** para consultar el modelo real.
   - Verifique que la **URL del Backend** sea `http://localhost:8000` (o `http://api:8000` si corre dentro de Docker).
3. En la pestaña **Calculadora de Riesgo**, ingrese los datos de prueba:
   - *Diabetes Mellitus*: `No`
   - *Preeclampsia*: `No`
   - *Edad Gestacional*: `32.0` semanas
   - *Peso al Nacer*: `1500` gramos
   - *APGAR a los 5 minutos*: `8`
4. Seleccione **⚡ Calcular Riesgo Clínico**.
5. Verifique la etiqueta de confirmación en pantalla:
   > **`🟢 Inferencia procesada por la API en vivo.`**
   *(Si la API no estuviera disponible, el sistema conmutaría automáticamente a Modo Mock con una alerta informativa).*

---

## 11. Detener los Servicios

Para detener los contenedores locales sin perder datos:

```bash
docker compose -f deploy/docker-compose.yml down
```

> **Nota sobre EC2**: Detener los contenedores locales no afecta la instancia en AWS. La instancia EC2 y el servidor MLflow permanecerán en ejecución salvo que se detengan explícitamente desde la consola de AWS para ahorro de recursos.

---

## 12. Ejecución Local Alternativa (Sin Docker, con uv)

Si desea ejecutar el proyecto directamente en su entorno local de desarrollo:

1. Cree el entorno virtual e instale dependencias:
   ```bash
   uv venv .venv --python 3.12
   uv pip install -e ".[dev]"
   ```

2. Ejecute las pruebas unitarias y linters:
   ```bash
   uv run poe check
   uv run poe test
   ```

3. Inicie los servicios en terminales independientes:
   ```bash
   # Terminal 1: Backend FastAPI
   uv run poe dev-api

   # Terminal 2: Frontend Streamlit
   uv run poe dev-dashboard
   ```

---

## 13. Buenas Prácticas y Seguridad

- **No versionar secretos**: Mantenga siempre fuera de control de versiones los archivos `.env`, `.dvc/config.local`, credenciales de AWS y archivos de llaves privadas (`.pem`).
- **Persistencia de modelos**: Todos los modelos deben ser logueados a través de `src/models/train.py` para garantizar su trazabilidad, registro de hiperparámetros y almacenamiento del artefacto `.joblib` en el Tracking Server de EC2.
- **Responsabilidad Clínica**: El puntaje y la clasificación generados por NeuroRisk constituyen una herramienta de soporte para la priorización del tamizaje; no sustituyen la valoración ni el diagnóstico médico profesional.
