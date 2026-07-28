class Docente:

    def __init__(
        self,
        nombre: str,
        especialidad: str,
        email: str,
        direccion: str = None,
        telefono: str = None,
        id: int = None,
    ):
        self.id = id
        self.nombre = nombre
        self.especialidad = especialidad
        self.email = email
        self.direccion = direccion
        self.telefono = telefono

    @classmethod
    def desde_tupla(cls, tupla):
        """Crea una instancia de Docente a partir de una tupla de la BD."""
        if not tupla:
            return None
        return cls(
            id=tupla[0],
            nombre=tupla[1],
            especialidad=tupla[2],
            email=tupla[3],
            direccion=tupla[4],
            telefono=tupla[5],
        )

    def a_diccionario(self):
        """Devuelve los datos del docente como diccionario."""
        return {
            "id": self.id,
            "nombre": self.nombre,
            "especialidad": self.especialidad,
            "email": self.email,
            "direccion": self.direccion,
            "telefono": self.telefono,
        }

    def __repr__(self):
        return f"<Docente id={self.id} nombre='{self.nombre}' especialidad='{self.especialidad}'>"