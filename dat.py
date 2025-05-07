import pandas as pd
import numpy as np

# Crear datos simulados
np.random.seed(42)  # Para reproducibilidad
num_transacciones = 500

# Generación de datos aleatorios
data = {
    "ID_Transaccion": np.arange(1, num_transacciones + 1),
    "Monto": np.random.randint(10, 10000, num_transacciones),  # Monto en dólares
    "Ubicación": np.random.choice(["Nueva York", "Londres", "Tokio", "Buenos Aires", "Madrid"], num_transacciones),
    "Método_Pago": np.random.choice(["Tarjeta Crédito", "Tarjeta Débito", "Transferencia", "Criptomoneda"], num_transacciones),
    "Hora_Transaccion": np.random.randint(0, 24, num_transacciones),  # Hora del día
    "Intentos_Fallidos": np.random.randint(0, 5, num_transacciones),  # Número de intentos antes de éxito
    "Fraude": np.random.choice([0, 1], num_transacciones, p=[0.85, 0.15])  # 15% fraude, 85% normal
}

# Crear DataFrame
df_transacciones = pd.DataFrame(data)

# Guardar en un archivo CSV
df_transacciones.to_csv("transacciones_bancarias.csv", index=False)

print(df_transacciones.head())  # Mostrar primeros registros
