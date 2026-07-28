from docente import Docente


class DocenteDAO:

    def __init__(self, db):
        self.db = db

    def crear_tabla(self):
        query = """
        CREATE TABLE IF NOT EXISTS docentes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            especialidad TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            direccion TEXT,
            telefono TEXT
        )
        """
        self.db.cursor.execute(query)
        self.db.conexion.commit()

    
    def insertar(
        self, nombre, especialidad, email, direccion=None, telefono=None
    ):
        query = """
        INSERT INTO docentes (nombre, especialidad, email, direccion, telefono) 
        VALUES (?, ?, ?, ?, ?)
        """
        self.db.cursor.execute(
            query, (nombre, especialidad, email, direccion, telefono)
        )
        self.db.conexion.commit()
        return self.db.cursor.lastrowid

    
    def insertar_registro(
        self, nombre, especialidad, email, direccion=None, telefono=None
    ):
        """Llama internamente a insertar para guardar un registro directamente."""
        return self.insertar(nombre, especialidad, email, direccion, telefono)

   
    def insertar_docente(self, docente: Docente):
        nuevo_id = self.insertar(
            nombre=docente.nombre,
            especialidad=docente.especialidad,
            email=docente.email,
            direccion=docente.direccion,
            telefono=docente.telefono,
        )
        docente.id = nuevo_id
        return nuevo_id

    def obtener_todos(self):
        query = "SELECT id, nombre, especialidad, email, direccion, telefono FROM docentes"
        self.db.cursor.execute(query)
        return self.db.cursor.fetchall()

    def obtener_por_id(self, docente_id):
        query = "SELECT id, nombre, especialidad, email, direccion, telefono FROM docentes WHERE id = ?"
        self.db.cursor.execute(query, (docente_id,))
        return self.db.cursor.fetchone()

    def actualizar(
        self,
        docente_id,
        nombre,
        especialidad,
        email,
        direccion=None,
        telefono=None,
    ):
        query = """
        UPDATE docentes 
        SET nombre = ?, especialidad = ?, email = ?, direccion = ?, telefono = ? 
        WHERE id = ?
        """
        self.db.cursor.execute(
            query,
            (nombre, especialidad, email, direccion, telefono, docente_id),
        )
        self.db.conexion.commit()

    def eliminar(self, docente_id):
        query = "DELETE FROM docentes WHERE id = ?"
        self.db.cursor.execute(query, (docente_id,))
        self.db.conexion.commit()