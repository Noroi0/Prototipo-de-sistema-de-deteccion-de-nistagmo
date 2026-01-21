# app.py
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pathlib
import shutil
import os
from passlib.context import CryptContext
import time
from keras.models import load_model
import cv2
import numpy as np
import traceback

# ---------- Config base ----------
APP_DIR = pathlib.Path(__file__).parent.resolve()

# Ruta al modelo CNN-LSTM
MODEL_PATH = APP_DIR / "modelo_nistagmo_CNN_LSTM.h5"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "nistagmo_tt.sqlite")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DB_PATH = "nistagmo_tt.sqlite"

CLASSES = ["no_nistagmo", "leve", "moderado", "severo"]

MENSAJES = {
    "no_nistagmo": " Felicidades, no se detectó nistagmo",
    "leve": " Nistagmo leve detectado",
    "moderado": " Nistagmo moderado detectado",
    "severo": " Nistagmo severo detectado"
}

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

model = load_model(str(MODEL_PATH))
print("Modelo CNN-LSTM cargado correctamente:", MODEL_PATH)

# Rutas de BD y uploads
SQLITE_PATH = APP_DIR / "nistagmo_tt.sqlite"   # archivo SQLite local (portátil)
UPLOADS_DIR = APP_DIR / "uploads"
INDEX_FILE = APP_DIR / "index.html"

# Usamos pbkdf2_sha256 para el hash (compatible con passlib)
pwd_ctx = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

app = FastAPI(title="SIDENI Demo API")

# monta carpeta static si existe (poner logo adentro como logo-sideni2.png)
if (APP_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

# CORS para desarrollo local (ajustar en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOADS_DIR, exist_ok=True)

# ---------- DB helpers ----------
def get_conn():
    # check_same_thread False para permitir uso en distintos hilos (uvicorn reload)
    conn = sqlite3.connect(str(SQLITE_PATH), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Inicializa las tablas según tu esquema. Si ya existen, no se tocan.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS usuario (
        id_usu INTEGER PRIMARY KEY AUTOINCREMENT,
        correo TEXT NOT NULL UNIQUE,
        contrasena_hash TEXT NOT NULL,
        creado_en DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS doctor (
        id_doc INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL,
        genero TEXT,
        ocupacion TEXT,
        nombre_clinica_hospital TEXT,
        usuario_id INTEGER,
        creado_en DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(usuario_id) REFERENCES usuario(id_usu) ON DELETE CASCADE
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS historial_paciente (
        id_pac INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL,
        edad INTEGER,
        genero TEXT,
        creado_en DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS doctor_historialPaciente (
        id_doc INTEGER NOT NULL,
        id_pac INTEGER NOT NULL,
        PRIMARY KEY(id_doc, id_pac),
        FOREIGN KEY(id_doc) REFERENCES doctor(id_doc),
        FOREIGN KEY(id_pac) REFERENCES historial_paciente(id_pac) ON DELETE CASCADE
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS registro_sesiones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        usuario_id INTEGER,
        fecha_hora DATETIME DEFAULT CURRENT_TIMESTAMP,
        ip TEXT,
        estado TEXT,
        FOREIGN KEY(usuario_id) REFERENCES usuario(id_usu)
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS video (
        id_video INTEGER PRIMARY KEY AUTOINCREMENT,
        paciente_id INTEGER,
        fecha_guardado DATETIME DEFAULT CURRENT_TIMESTAMP,
        ruta_video TEXT,
        FOREIGN KEY(paciente_id) REFERENCES historial_paciente(id_pac) ON DELETE CASCADE
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS resultado_clinico (
        id_resultado INTEGER PRIMARY KEY AUTOINCREMENT,
        id_pac INTEGER,
        id_video INTEGER,
        resultado TEXT CHECK(resultado IN ('severo','moderado','leve','no_nistagmo')),
        fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(id_pac) REFERENCES historial_paciente(id_pac),
        FOREIGN KEY(id_video) REFERENCES video(id_video)
    )""")
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup_event():
    init_db()

# ---------- Helpers de respuesta ----------
def json_ok(**kwargs):
    out = {"ok": True}
    out.update(kwargs)
    return JSONResponse(content=out)

def json_err(mensaje="Error interno", code=400):
    return JSONResponse(status_code=code, content={"ok": False, "mensaje": mensaje})

# ---------- Rutas de entrega de front ----------
@app.get("/", response_class=FileResponse)
async def serve_index():
    return FileResponse(INDEX_FILE)

# ---------- API: crear usuario ----------
@app.post("/api/usuarios")
async def create_usuario(request: Request):
    """
    Recibe JSON: { "email": "...", "password": "..." }
    - Convierte password -> hash (pbkdf2_sha256)
    - Inserta en tabla 'usuario' (correo, contrasena_hash)
    Respuesta esperada:
      - éxito: { ok:true, mensaje: "Cuenta creada correctamente" }
      - fallo:  { ok:false, mensaje: "..." }
    """
    try:
        payload = await request.json()
    except Exception:
        return json_err("Carga inválida (JSON).", 422)

    correo = payload.get("email") or payload.get("correo")
    password = payload.get("password") or payload.get("contrasena")
    if not correo or not password:
        return json_err("Faltan campos (correo/contraseña).", 400)

    hashed = pwd_ctx.hash(password)
    try:
        conn = get_conn(); cur = conn.cursor()
        cur.execute("INSERT INTO usuario (correo, contrasena_hash) VALUES (?, ?)", (correo, hashed))
        conn.commit(); conn.close()
        return json_ok(mensaje="Cuenta creada correctamente")
    except sqlite3.IntegrityError:
        return json_err("El correo ya está registrado.", 409)
    except Exception as e:
        return json_err(f"Error al crear usuario: {str(e)}", 500)

# ---------- API: login ----------
@app.post("/api/login")
async def login(request: Request):
    """
    Recibe JSON: { "email": "...", "password": "..." }
    - Busca usuario por correo
    - Verifica password con passlib.verify
    - Si OK: inserta una fila en registro_sesiones con usuario_id, ip y estado='login'
    - Responde: { ok:true, usuario: { id_usu, correo } }
    NOTAS IMPORTANTES PARA EL INTEGRADOR:
      - Si el hash almacenado NO es pbkdf2_sha256 (p ej: bcrypt), passlib puede
        tirar UnknownHashError. Si tu compañero usa otra librería para generar
        hashes, asegúrate de usar la misma en app.py o de normalizar hashes.
    """
    try:
        payload = await request.json()
    except Exception:
        return json_err("Carga inválida (JSON).", 422)

    correo = payload.get("email") or payload.get("correo")
    password = payload.get("password") or payload.get("contrasena")
    if not correo or not password:
        return json_err("Faltan campos (correo/contraseña).", 400)

    try:
        conn = get_conn(); cur = conn.cursor()
        cur.execute("SELECT id_usu, correo, contrasena_hash FROM usuario WHERE correo = ?", (correo,))
        row = cur.fetchone()
        if not row:
            # No registrado
            return JSONResponse(content={"ok": False, "mensaje": "Usuario no registrado"})
        contr_hash = row["contrasena_hash"]
        try:
            ok = pwd_ctx.verify(password, contr_hash)
        except Exception:
            # si el hash es de un esquema no soportado o corrupto
            return JSONResponse(content={"ok": False, "mensaje": "Contraseña incorrecta"})
        if not ok:
            return JSONResponse(content={"ok": False, "mensaje": "Contraseña incorrecta"})
        # Login correcto -> registrar sesión
        client = request.client
        ip = client.host if client else None
        cur.execute("INSERT INTO registro_sesiones (usuario_id, ip, estado) VALUES (?, ?, ?)", (row["id_usu"], ip, "login"))
        conn.commit(); conn.close()
        return JSONResponse(content={"ok": True, "usuario": {"id_usu": row["id_usu"], "correo": row["correo"]}})
    except Exception as e:
        return json_err(f"Error interno: {str(e)}", 500)

# ---------- API: crear doctor ----------
@app.post("/api/doctores")
async def create_doctor(request: Request):
    """
    Recibe JSON con: nombre, genero, ocupacion, clinica, usuario_email (o usuario_id)
    - Busca usuario por correo para obtener usuario_id (si se pasa correo)
    - Inserta fila en 'doctor' con usuario_id (vínculo)
    Respuesta esperada: { ok:true, mensaje: "Doctor guardado correctamente" }
    """
    try:
        payload = await request.json()
    except Exception:
        return json_err("Carga inválida (JSON).", 422)

    nombre = payload.get("nombre")
    genero = payload.get("genero")
    ocupacion = payload.get("ocupacion")
    clinica = payload.get("clinica") or payload.get("nombre_clinica_hospital")
    usuario_email = payload.get("usuario_email")
    usuario_id = payload.get("usuario_id")

    if not usuario_email and not usuario_id:
        return json_err("Usuario no autenticado", 401)

    try:
        conn = get_conn(); cur = conn.cursor()
        if usuario_email:
            cur.execute("SELECT id_usu FROM usuario WHERE correo = ?", (usuario_email,))
            r = cur.fetchone()
            if not r:
                return json_err("Usuario no encontrado", 404)
            usuario_id = r["id_usu"]
        cur.execute("""INSERT INTO doctor (nombre, genero, ocupacion, nombre_clinica_hospital, usuario_id)
                       VALUES (?, ?, ?, ?, ?)""", (nombre, genero, ocupacion, clinica, usuario_id))
        conn.commit(); conn.close()
        return json_ok(mensaje="Doctor guardado correctamente")
    except Exception as e:
        return json_err(f"Error al guardar doctor: {str(e)}", 500)

# ---------- API: crear paciente ----------
@app.post("/api/pacientes")
async def create_paciente(request: Request):
    """
    Recibe JSON: { nombre, edad, genero, usuario_email (opcional) }
    - Inserta fila en historial_paciente y devuelve paciente_id
    - Si se quiere asociar doctor <-> paciente, backend espera id_doc en payload y
      puede insertar en doctor_historialPaciente (línea comentada)
    Respuesta esperada: { ok:true, paciente_id: <int>, mensaje: "..."}
    """
    try:
        payload = await request.json()
    except Exception:
        return json_err("Carga inválida (JSON).", 422)

    nombre = payload.get("nombre")
    edad = payload.get("edad")
    genero = payload.get("genero")
    if not nombre:
        return json_err("Falta nombre", 400)
    try:
        conn = get_conn(); cur = conn.cursor()
        cur.execute("INSERT INTO historial_paciente (nombre, edad, genero) VALUES (?,?,?)", (nombre, edad, genero))
        paciente_id = cur.lastrowid
        # Si viene id_doc: asociar (descomentar si se usará)
        # if payload.get("id_doc"): cur.execute("INSERT OR IGNORE INTO doctor_historialPaciente (id_doc,id_pac) VALUES (?,?)", (payload['id_doc'], paciente_id))
        conn.commit(); conn.close()
        return json_ok(mensaje="Paciente registrado correctamente", paciente_id=paciente_id)
    except Exception as e:
        return json_err(f"Error al guardar paciente: {str(e)}", 500)

# ---------- API: subir video (multipart/form-data) ----------
@app.post("/api/upload_video")
async def upload_video(paciente_id: int = Form(...), file: UploadFile = File(...)):
    """
    Endpoint de subida:
    - Entrada: form-data con campo 'paciente_id' (int) y 'file' (archivo de video)
    - Acción: guardar archivo en ./uploads con nombre único y crear registro en tabla 'video'
    - Respuesta esperada: { ok:true, video_id: <int>, ruta: "<nombre_archivo>" }
    NOTAS:
      - Archivo guardado con nombre único: video_<paciente>_<timestamp>.ext
      - El campo ruta_video en BD guarda solo el nombre (o ruta relativa)
    """
    try:
        ext = pathlib.Path(file.filename).suffix or ".mp4"
        timestamp = int(time.time() * 1000)
        dest_name = f"video_{paciente_id}_{timestamp}{ext}"
        dest_path = UPLOADS_DIR / dest_name
        with dest_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        conn = get_conn(); cur = conn.cursor()
        cur.execute("INSERT INTO video (paciente_id, ruta_video) VALUES (?,?)", (paciente_id, str(dest_name)))
        video_id = cur.lastrowid
        conn.commit(); conn.close()
        return json_ok(mensaje="Video subido", video_id=video_id, ruta=str(dest_name))
    except Exception as e:
        return json_err(f"Error al subir video: {str(e)}", 500)

def load_video_for_model(video_path, max_frames=40, size=(128,128)):
    import cv2
    frames = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    i = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if i % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, size)
            gray = gray / 255.0
            frames.append(gray)

        i += 1

    cap.release()

    # padding si faltan frames
    while len(frames) < max_frames:
        frames.append(frames[-1])

    X = np.array(frames, dtype=np.float32)
    X = X[..., np.newaxis]     # (40,128,128,1)
    X = np.expand_dims(X, 0)   # (1,40,128,128,1)
    return X


# ---------- API: analyze_video (INTEGRADO CON MODELO) ----------
# ---------- API: analyze_video (INTEGRADO CON MODELO) ----------
@app.post("/api/analyze_video")
async def analyze_video(request: Request):
    try:
        data = await request.json()
        paciente_id = data.get("paciente_id")
        video_id = data.get("video_id")
        
        # Obtener la ruta del video desde la BD usando video_id
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT ruta_video FROM video WHERE id_video = ?", (video_id,))
        video_row = cur.fetchone()
        
        if not video_row:
            return json_err("Video no encontrado en BD", 404)
        
        ruta = video_row["ruta_video"]
        conn.close()
        
        if not paciente_id or not ruta:
            return json_err("Faltan datos para análisis", 400)

        video_path = UPLOADS_DIR / ruta
        if not video_path.exists():
            return json_err("Video no encontrado en servidor", 404)

        # -------- LEER VIDEO --------
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while len(frames) < 40:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (128, 128))
            frames.append(frame)

        cap.release()

        if len(frames) < 40:
            return json_err("Video demasiado corto (se necesitan al menos 40 frames)", 400)

        X = np.array(frames) / 255.0
        X = X.reshape(1, 40, 128, 128, 1)

        # -------- INFERENCIA --------
        pred = model.predict(X, verbose=0)[0]
        clase_idx = int(np.argmax(pred))
        score = float(pred[clase_idx])

        labels = ["no_nistagmo", "leve", "moderado", "severo"]
        clase = labels[clase_idx]

        etiquetas = {
            "no_nistagmo": "Felicidades, no se detectó nistagmo",
            "leve": " Nistagmo leve detectado",
            "moderado": " Nistagmo moderado detectado",
            "severo": " Nistagmo severo detectado"
        }

        diagnostico = clase

        # -------- GUARDAR EN BD --------
        conn = get_conn()
        cur = conn.cursor()

        # Normalizar etiqueta para BD
        if diagnostico == "no_nistagmo":
            diagnostico_db = "no_presencia"
        else:
            diagnostico_db = diagnostico

        # Insertar en resultado_clinico
            cur.execute("""
                INSERT INTO resultado_clinico (id_pac, id_video, resultado)
                VALUES (?, ?, ?)
            """, (paciente_id, video_id, diagnostico_db))

        # También actualizar tabla pacientes si existe
        try:
            cur.execute("""
                UPDATE pacientes SET diagnostico=? WHERE id=?
            """, (diagnostico, paciente_id))
        except:
            pass  # Ignorar si la tabla no existe
        
        conn.commit()
        conn.close()

        return json_ok(
            diagnostico=diagnostico,
            mensaje=etiquetas.get(diagnostico, "Análisis completado"),
            score=round(score, 3)
        )

    except Exception as e:
        traceback.print_exc()
        return json_err(f"Error interno: {str(e)}", 500)


@app.get("/api/doctores")
async def list_doctores():
    try:
        conn = get_conn(); cur = conn.cursor()
        cur.execute("SELECT * FROM doctor")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        return json_err(str(e), 500)

@app.get("/api/pacientes")
async def list_pacientes():
    try:
        conn = get_conn(); cur = conn.cursor()
        cur.execute("SELECT * FROM historial_paciente")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        return json_err(str(e), 500)

@app.get("/api/videos")
async def list_videos():
    try:
        conn = get_conn(); cur = conn.cursor()
        cur.execute("SELECT * FROM video")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        return json_err(str(e), 500)