import sqlite3


class ConexionDB:

    def __init__(self, db_name="escuela.db"):
        self.conexion = sqlite3.connect(db_name)
        self.cursor = self.conexion.cursor()

    def cerrar(self):
        self.conexion.close()


# --- Prueba de ejecución ---
db = ConexionDB()
docente_dao = DocenteDAO(db)

# 1. Crear tabla
docente_dao.crear_tabla()

# 2. Insertar docentes
docente_dao.insertar("Carlos Mendoza", "Matemáticas", "carlos@escuela.com")
docente_dao.insertar("Laura Gómez", "Programación", "laura@escuela.com")

# 3. Consultar todos los docentes
print("--- Lista de Docentes ---")
for docente in docente_dao.obtener_todos():
    print(f"ID: {docente[0]} | Nombre: {docente[1]} | Especialidad: {docente[2]}")

db.cerrar()