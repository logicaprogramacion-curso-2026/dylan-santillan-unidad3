import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ============ ESTRUCTURAS DE DATOS ============
# Tuplas: catálogo de productos (inmutable)
CATALOGO = (
    (1, "Laptop", "Electrónica"),
    (2, "Mouse", "Accesorios"),
    (3, "Teclado", "Accesorios"),
    (4, "Monitor", "Electrónica"),
    (5, "Auriculares", "Audio"),
    (6, "Webcam", "Accesorios")
)

# Diccionarios: precios y stock
PRECIOS = {
    1: 12000.00,
    2: 350.50,
    3: 800.00,
    4: 4500.00,
    5: 1500.75,
    6: 900.00
}

STOCK = {
    1: 10,
    2: 50,
    3: 30,
    4: 15,
    5: 25,
    6: 20
}

# Listas: buffer de ventas
ventas_buffer = []
ventas_ids = []

# ============ FUNCIONES ============
def mostrar_catalogo():
    """Muestra el catálogo de productos"""
    print("\n=== CATÁLOGO DE PRODUCTOS ===")
    print("ID | Producto | Categoría | Precio | Stock")
    print("-" * 50)
    for producto in CATALOGO:
        id_prod, nombre, categoria = producto
        print(f"{id_prod} | {nombre} | {categoria} | ${PRECIOS[id_prod]:.2f} | {STOCK[id_prod]}")
    print()

def registrar_venta():
    """Registra una nueva venta"""
    try:
        mostrar_catalogo()
        producto_id = int(input("Ingrese el ID del producto: "))
        
        # Validar que el producto exista
        if producto_id not in PRECIOS:
            # Reto D: Registrar intento fallido en log
            with open("log.txt", "a") as log:
                log.write(f"{datetime.now()}: Intento fallido de venta - Producto ID {producto_id} no existe\n")
            print("❌ Producto no encontrado en el catálogo.")
            return
        
        unidades = int(input("Ingrese la cantidad de unidades: "))
        
        # Validar stock disponible
        if unidades > STOCK[producto_id]:
            print(f"❌ Stock insuficiente. Solo hay {STOCK[producto_id]} unidades disponibles.")
            return
        
        if unidades <= 0:
            print("❌ La cantidad debe ser mayor a cero.")
            return
        
        # Calcular precio con descuento
        precio_unitario = PRECIOS[producto_id]
        subtotal = precio_unitario * unidades
        
        # Reto C: Aplicar descuento si unidades >= 10
        descuento = 0
        if unidades >= 10:
            descuento = subtotal * 0.05  # 5% de descuento
            subtotal = subtotal - descuento
            print(f"🎉 Descuento aplicado: 5% (-${descuento:.2f})")
        
        # Actualizar stock
        STOCK[producto_id] -= unidades
        
        # Encontrar el nombre del producto
        nombre_producto = ""
        for prod in CATALOGO:
            if prod[0] == producto_id:
                nombre_producto = prod[1]
                break
        
        # Registrar venta
        venta = {
            'id_venta': len(ventas_buffer) + 1,
            'producto_id': producto_id,
            'producto': nombre_producto,
            'unidades': unidades,
            'precio_unitario': precio_unitario,
            'subtotal': subtotal,
            'descuento': descuento,
            'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        ventas_buffer.append(venta)
        ventas_ids.append(venta['id_venta'])
        
        print(f"✅ Venta registrada: {unidades}x {venta['producto']} - Total: ${subtotal:.2f}")
        
        # Guardar en log
        with open("log.txt", "a") as log:
            log.write(f"{datetime.now()}: Venta exitosa - ID: {venta['id_venta']}, "
                     f"Producto: {venta['producto']}, Unidades: {unidades}, Total: ${subtotal:.2f}\n")
        
    except ValueError:
        print("❌ Error: Debe ingresar un número válido.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

def guardar_ventas_csv():
    """Guarda las ventas en un archivo CSV"""
    try:
        if not ventas_buffer:
            print("⚠️ No hay ventas para guardar.")
            return
        
        df = pd.DataFrame(ventas_buffer)
        df.to_csv("ventas.csv", index=False)
        print(f"✅ {len(ventas_buffer)} ventas guardadas en ventas.csv")
        
    except Exception as e:
        print(f"❌ Error al guardar CSV: {e}")

def analizar_ventas():
    """Analiza las ventas usando Pandas y NumPy"""
    try:
        # Intentar leer CSV existente
        if os.path.exists("ventas.csv"):
            df = pd.read_csv("ventas.csv")
        else:
            if not ventas_buffer:
                print("⚠️ No hay datos de ventas.")
                return
            df = pd.DataFrame(ventas_buffer)
        
        if df.empty:
            print("⚠️ No hay datos de ventas para analizar.")
            return
        
        print("\n=== ANÁLISIS DE VENTAS ===")
        
        # Pandas: Groupby para análisis
        ventas_por_producto = df.groupby('producto').agg({
            'unidades': 'sum',
            'subtotal': 'sum'
        }).reset_index()
        
        print("\n📊 Ventas por producto:")
        print(ventas_por_producto.to_string(index=False))
        
        # NumPy: Cálculos estadísticos
        subtotales = df['subtotal'].values
        unidades = df['unidades'].values
        
        media_ventas = np.mean(subtotales)
        desviacion = np.std(subtotales)
        total_ingresos = np.sum(subtotales)
        total_unidades = np.sum(unidades)
        
        print(f"\n📈 Estadísticas:")
        print(f"Total de ingresos: ${total_ingresos:.2f}")
        print(f"Total de unidades vendidas: {total_unidades}")
        print(f"Promedio por venta: ${media_ventas:.2f}")
        print(f"Desviación estándar: ${desviacion:.2f}")
        
        # Detectar producto más vendido
        producto_top = ventas_por_producto.loc[ventas_por_producto['subtotal'].idxmax()]
        print(f"\n🏆 Producto con más ingresos: {producto_top['producto']} (${producto_top['subtotal']:.2f})")
        
    except FileNotFoundError:
        print("❌ Archivo ventas.csv no encontrado.")
    except ZeroDivisionError:
        print("❌ No se puede calcular la media: no hay ventas registradas.")
    except Exception as e:
        print(f"❌ Error en el análisis: {e}")

def graficar_ingresos():
    """Genera gráfico de ingresos por producto"""
    try:
        # Leer datos
        if os.path.exists("ventas.csv"):
            df = pd.read_csv("ventas.csv")
        elif ventas_buffer:
            df = pd.DataFrame(ventas_buffer)
        else:
            print("⚠️ No hay datos para graficar.")
            return
        
        if df.empty:
            print("⚠️ No hay datos de ventas.")
            return
        
        # Agrupar por producto
        ventas_por_producto = df.groupby('producto')['subtotal'].sum()
        
        # Crear gráfico
        plt.figure(figsize=(10, 6))
        plt.bar(ventas_por_producto.index, ventas_por_producto.values, color='skyblue')
        plt.title('Ingresos por Producto', fontsize=14, fontweight='bold')
        plt.xlabel('Producto', fontsize=12)
        plt.ylabel('Ingresos ($)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Mostrar gráfico
        plt.show()
        
        # Reto B: Exportar a PNG
        respuesta = input("¿Desea exportar el gráfico a PNG? (s/n): ").lower()
        if respuesta == 's':
            plt.savefig("ingresos.png", dpi=300, bbox_inches='tight')
            print("✅ Gráfico guardado como ingresos.png")
        
    except Exception as e:
        print(f"❌ Error al generar gráfico: {e}")

def agregar_producto():
    """Reto A: Agregar nuevo producto al catálogo"""
    global CATALOGO  # Declarar global al inicio de la función
    
    try:
        print("\n=== AGREGAR NUEVO PRODUCTO ===")
        nuevo_id = max(PRECIOS.keys()) + 1
        nombre = input("Nombre del producto: ")
        categoria = input("Categoría: ")
        precio = float(input("Precio: $"))
        stock = int(input("Stock inicial: "))
        
        # Agregar a las estructuras
        nuevo_producto = (nuevo_id, nombre, categoria)
        
        # Convertir tupla a lista, agregar elemento, volver a tupla
        catalogo_lista = list(CATALOGO)
        catalogo_lista.append(nuevo_producto)
        CATALOGO = tuple(catalogo_lista)
        
        # Actualizar diccionarios
        PRECIOS[nuevo_id] = precio
        STOCK[nuevo_id] = stock
        
        print(f"✅ Producto '{nombre}' agregado con ID {nuevo_id}")
        
        # Log
        with open("log.txt", "a") as log:
            log.write(f"{datetime.now()}: Producto agregado - {nombre} (ID: {nuevo_id})\n")
        
    except ValueError:
        print("❌ Error: Datos inválidos.")
    except Exception as e:
        print(f"❌ Error: {e}")

def mostrar_menu():
    """Muestra el menú principal"""
    print("\n" + "="*40)
    print("🏪 MINITIENDA - SISTEMA DE VENTAS")
    print("="*40)
    print("1. Registrar venta")
    print("2. Guardar ventas en CSV")
    print("3. Analizar ventas")
    print("4. Graficar ingresos por producto")
    print("5. Agregar nuevo producto")
    print("6. Salir")
    print("="*40)

def generar_datos_prueba():
    """Genera datos de prueba para demostración"""
    datos_prueba = [
        (1, 3), (2, 5), (3, 2), (4, 1), (5, 4),
        (1, 2), (2, 10), (3, 5), (6, 3), (4, 2),
        (5, 8), (1, 4)
    ]
    
    for prod_id, unidades in datos_prueba:
        if prod_id in PRECIOS and STOCK[prod_id] >= unidades:
            precio_unitario = PRECIOS[prod_id]
            subtotal = precio_unitario * unidades
            descuento = 0
            
            if unidades >= 10:
                descuento = subtotal * 0.05
                subtotal -= descuento
            
            # Encontrar nombre del producto
            nombre_producto = ""
            for prod in CATALOGO:
                if prod[0] == prod_id:
                    nombre_producto = prod[1]
                    break
            
            venta = {
                'id_venta': len(ventas_buffer) + 1,
                'producto_id': prod_id,
                'producto': nombre_producto,
                'unidades': unidades,
                'precio_unitario': precio_unitario,
                'subtotal': subtotal,
                'descuento': descuento,
                'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            ventas_buffer.append(venta)
            STOCK[prod_id] -= unidades
    
    print(f"✅ {len(datos_prueba)} ventas de prueba generadas")

def main():
    """Función principal con bucle while"""
    print("🏪 Bienvenido a MiniTienda")
    print("Sistema de registro y análisis de ventas\n")
    
    # Preguntar si quiere generar datos de prueba
    respuesta = input("¿Desea generar datos de prueba? (s/n): ").lower()
    if respuesta == 's':
        generar_datos_prueba()
        guardar_ventas_csv()
    
    while True:
        try:
            mostrar_menu()
            opcion = input("Seleccione una opción (1-6): ").strip()
            
            if opcion == '1':
                registrar_venta()
            elif opcion == '2':
                guardar_ventas_csv()
            elif opcion == '3':
                analizar_ventas()
            elif opcion == '4':
                graficar_ingresos()
            elif opcion == '5':
                agregar_producto()
            elif opcion == '6':
                print("👋 ¡Gracias por usar MiniTienda!")
                # Guardar automáticamente antes de salir
                if ventas_buffer:
                    guardar_ventas_csv()
                break
            else:
                print("❌ Opción inválida. Intente nuevamente.")
                continue
                
        except KeyboardInterrupt:
            print("\n\n👋 Saliendo del programa...")
            break
        except EOFError:
            print("\n👋 Saliendo del programa...")
            break
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            continue
        finally:
            # Este bloque se ejecuta siempre
            pass

# ============ EJECUCIÓN ============
if __name__ == "__main__":
    # Limpiar archivos si existen
    if os.path.exists("log.txt"):
        os.remove("log.txt")
    
    # Ejecutar programa
    main()