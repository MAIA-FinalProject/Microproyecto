# Manual de Usuario — NeuroRisk

## 1. Propósito de la herramienta

NeuroRisk es un tablero de apoyo para estimar tempranamente el riesgo de alteración del neurodesarrollo en neonatos prematuros.

La herramienta utiliza información materna y neonatal para generar:

- Un puntaje de riesgo entre 0 y 100.
- Una categoría cualitativa: Bajo, Moderado, Alto o Crítico.
- Una recomendación de seguimiento clínico.

El resultado es un apoyo para priorizar el seguimiento. No reemplaza la valoración médica, el diagnóstico ni el juicio clínico profesional.

## 2. Acceso al tablero

Cuando la aplicación se ejecuta localmente, abra en el navegador:

```text
http://localhost:8501
```

El tablero presenta dos módulos principales:

1. **Calculadora de Riesgo**: permite evaluar un neonato.
2. **Exploración Clínica**: permite consultar información descriptiva de la cohorte.

## 3. Configuración inicial

En la barra lateral encontrará la sección **Configuración**.

### Modo Simulación (Mock API)

El interruptor **Modo Simulación (Mock API)** permite elegir la fuente del resultado:

- **Activado**: genera una estimación simulada directamente en el tablero. Es útil para demostraciones o cuando la API no está disponible.
- **Desactivado**: envía los datos al backend FastAPI y utiliza el modelo entrenado.

En modo live, confirme que la **URL del Backend (FastAPI)** sea correcta. Para una ejecución local normalmente es:

```text
http://localhost:8000
```

El resultado indica si fue generado por el modo Mock o por la API en vivo. Si la API no responde, el tablero muestra una advertencia y utiliza el modo Mock como respaldo.

## 4. Realizar una evaluación de riesgo

Entre en la pestaña **Calculadora de Riesgo** y complete el formulario.

### 4.1 Factores maternos

Seleccione una opción para cada variable:

- **Diabetes Mellitus Materna (DM)**: indique si la madre tenía diabetes mellitus diagnosticada.
- **Preeclampsia durante el embarazo**: indique si se presentó preeclampsia.

### 4.2 Factores neonatales

Ingrese o seleccione:

- **Edad Gestacional**: semanas de gestación al momento del parto.
- **Peso al Nacer**: peso registrado en gramos.
- **Puntaje APGAR a los 5 minutos**: valor entre 0 y 10.

Los rangos disponibles en el formulario son:

- Edad gestacional: 24 a 38 semanas.
- Peso al nacer: 500 a 3500 gramos.
- APGAR a los 5 minutos: 0 a 10.

### 4.3 Generar el resultado

Después de completar los campos:

1. Revise que la información corresponda al neonato evaluado.
2. Seleccione **Calcular Riesgo Clínico**.
3. Espere a que termine el procesamiento.
4. Revise el puntaje, la categoría, la recomendación y el modo utilizado.

## 5. Interpretación del resultado

El tablero muestra un puntaje de 0 a 100. Las categorías se asignan así:

| Puntaje | Categoría | Orientación mostrada por la herramienta |
|---|---|---|
| 0 a menor de 25 | Bajo | Continuar con el seguimiento neonatal estándar y el tamizaje rutinario. |
| 25 a menor de 50 | Moderado | Intensificar la monitorización y programar evaluación neuroconductual antes del egreso. |
| 50 a menor de 75 | Alto | Priorizar la valoración por neuropediatría, la evaluación ecográfica cerebral y el seguimiento en neurodesarrollo. |
| 75 a 100 | Crítico | Activar una alerta clínica prioritaria, intervención temprana, panel multidisciplinario y neuroimagen. |

La recomendación que aparece en pantalla debe interpretarse junto con la evaluación clínica completa del paciente.

## 6. Explorar la cohorte clínica

Seleccione la pestaña **Exploración Clínica** para consultar información descriptiva de los neonatos disponibles.

El tablero presenta:

- Total de neonatos en la cohorte.
- Porcentaje con alteración del neurodesarrollo.
- Edad gestacional promedio.
- Peso al nacer promedio.
- Distribución del resultado `neurodev_alteration`.
- Relación entre edad gestacional y peso al nacer.
- Una muestra de hasta 25 registros.

Esta sección sirve para contextualizar los resultados de la cohorte. No calcula el riesgo individual del paciente ingresado en la calculadora.

## 7. Limpiar una evaluación

Para borrar el resultado actual y comenzar una nueva evaluación:

1. En la barra lateral, seleccione **Limpiar Evaluación Actual**.
2. Confirme que el resultado desapareció de la calculadora.
3. Ingrese los datos del siguiente neonato.

## 8. Consideraciones importantes

- Verifique las unidades: la edad gestacional se registra en semanas y el peso al nacer en gramos.
- El APGAR corresponde al valor medido a los 5 minutos, no al valor del primer minuto.
- No interprete un puntaje como diagnóstico definitivo.
- Confirme si el resultado proviene del modelo en vivo o del modo Mock antes de utilizarlo en una demostración o análisis.
- Si el tablero indica que la API no está disponible, comuníquese con la persona responsable de la instalación antes de usar el resultado para una evaluación real.
