# Diccionario de Datos

Documentación del dataset crudo versionado con DVC en `data/raw/` (ver [raw.dvc](raw.dvc)).

- **Nombre:** Dataset on neonatal and maternal factors influencing neurodevelopmental outcomes in preterm infants.
- **Fuente:** Darabi, A., Faramarzi, R., Boskabadi, H., Maamouri, G., Rezvani, R. (2024). [Mendeley Data](https://data.mendeley.com/datasets/h464gsf77t/2) | [Artículo asociado](https://www.sciencedirect.com/science/article/pii/S2352340924000325).
- **Cohorte:** 89 neonatos prematuros, Hospital Ghaem (Mashhad, Irán), 2016–2020.
- **Dimensiones:** 89 filas × 53 columnas crudas (54 tras agregar el target compuesto en `load_labeled()`).
- **Formato original:** `.sav` (SPSS), leído vía `pyreadstat` en [src/data/loader.py](../src/data/loader.py).
- **Codebook original:** versionado en `data/raw/codebook.pdf` (trackeado por DVC junto con `preterm_infant.sav` y `preterm_infant.xlsx`). Las descripciones de abajo salen del código y del EDA ya corrido ([notebooks/EDA.ipynb](../notebooks/EDA.ipynb)); las variables marcadas como *pendiente* deben confirmarse contra ese PDF.

> Nota de alcance: esta tabla se armó sin tener `data/raw/` descargado localmente (pendiente `dvc pull`, ver sección de verificación DVC+S3). Los valores de nulos/rangos vienen de outputs ya guardados en el notebook de Bloque 3, no de una ejecución propia. Al hacer `dvc pull`, vale la pena re-verificar esta tabla contra el dataset completo (89 filas), no solo contra la muestra de 5 filas visible en `df.head()`.

---

## Variables maternas / del embarazo

| Variable | Tipo | Descripción | Valores observados | Notas |
|---|---|---|---|---|
| `DM` | categórica binaria | Diabetes mellitus materna | `yes` / `no` | |
| `preeclampsia` | categórica binaria | Preeclampsia durante el embarazo | `yes` / `no` | |
| `hypothyroid` | categórica binaria | Hipotiroidismo materno | `yes` / `no` | |
| `PROM` | categórica binaria | Ruptura prematura de membranas (*Premature Rupture Of Membranes*) | `yes` / `no` | |
| `IUGR` | categórica binaria | Restricción del crecimiento intrauterino (*Intrauterine Growth Restriction*) | `YES` / `NO` | **Casing inconsistente confirmado en el codebook fuente** (`1.00 = YES, 2.00 = NO`, en mayúsculas mientras el resto de binarias usan minúsculas) — no es un error del pipeline, hay que normalizar explícitamente antes de modelar |
| `pregnancycomplication` | categórica binaria | Presencia de alguna complicación del embarazo | `yes` / `no` | Bloque 4 (EDA enfocado) la está analizando en detalle |
| `drug.mother` | categórica binaria | Consumo de medicamentos/sustancias durante el embarazo | `yes` / `no` | Pendiente confirmar si incluye solo prescritos o también otros |
| `mother.sonogarphy.result` | categórica | Resultado de ecografía materna | `normal` / `abnormal` (a confirmar exhaustividad) | Nombre de columna con typo de origen (`sonogarphy`) — se conserva tal cual viene del dataset |
| `laborType` | categórica | Tipo de parto | `NVD` (parto vaginal normal), `cs` (cesárea) | Confirmar si existen más categorías |

## Complicaciones neonatales

| Variable | Tipo | Descripción | Valores observados | Notas |
|---|---|---|---|---|
| `pneumothorax` | categórica binaria | Neumotórax | `yes` / `no` | |
| `NEC` | categórica binaria | Enterocolitis necrotizante | `yes` / `no` | |
| `sepsis` | categórica binaria | Sepsis neonatal | `yes` / `no` | **Ya no es el target del proyecto** (pivote documentado en README raíz, sección 1.1) — desbalance severo, 5/89 positivos (5.6%). Se mantiene como predictor |
| `PDA` | categórica binaria | Conducto arterioso persistente (*Patent Ductus Arteriosus*) | `yes` / `no` | |
| `icter` | categórica binaria | Ictericia neonatal | `yes` / `no` | |
| `meningitis` | categórica binaria | Meningitis | `yes` / `no` | |
| `IVH` | categórica binaria | Hemorragia intraventricular (*Intraventricular Hemorrhage*) | `yes` / `no` | |
| `seizure` | categórica binaria | Convulsiones | `yes` / `no` | |
| `BPD` | categórica binaria | Displasia broncopulmonar (*Bronchopulmonary Dysplasia*) | `yes` / `no` | |
| `congenital.anomaly` | categórica binaria | Anomalía congénita | `yes` / `no` | |
| `B.C` | categórica | Hemocultivo (*Blood Culture*) | `positive` / `negative` (confirmado en codebook: `1.00 = positive, 2.00 = negative`) | Relacionada con `sepsis`/`SepsisnegativeCulture` — el codebook no documenta la relación clínica exacta entre las tres variables de cultivo, queda a criterio del equipo |
| `csf.culture` | categórica | Cultivo de líquido cefalorraquídeo | `Positive` / `Negatiev` (confirmado en codebook) | **El typo "Negatiev" viene del codebook original** (`1.00 = Positive, 2.00 = Negatiev`), no fue introducido por nuestro pipeline — conservar tal cual o normalizar con criterio explícito documentado |
| `SepsisnegativeCulture` | categórica binaria | Sepsis con cultivo negativo (sepsis clínica sin confirmación de laboratorio) | `yes` / `no` (confirmado en codebook) | El codebook solo documenta la codificación de valores, no da descripción clínica — semántica exacta vs. `sepsis`/`B.C` sigue siendo criterio de equipo |

## Intervenciones médicas

| Variable | Tipo | Descripción | Valores observados | Notas |
|---|---|---|---|---|
| `surfactant` | categórica binaria | Administración de surfactante pulmonar | `yes` / `no` | Bloque 4 la está verificando |
| `aggressive.ventilation` | categórica binaria | Ventilación agresiva | `yes` / `no` | Bloque 4 la está verificando |
| `notaggressive.ventilation` | categórica binaria | Ventilación no agresiva | `yes` / `no` | Confirmar si es mutuamente excluyente con `aggressive.ventilation` o pueden coexistir |
| `type.of.ressucitation` | categórica | Tipo de reanimación al nacer | `ppv` (presión positiva) / `advanced` (confirmado en codebook: `1.00 = ppv, 2.00 = advanced`) | **66% nulos (59/89)** — probablemente no-aleatorio (solo se registra si hubo reanimación); candidata a categoría propia `"no_resuscitation"` en vez de imputación |
| `ehya.badve.tavallod` | categórica binaria | Sin confirmar — el codebook solo documenta la codificación (`1.00 = yes, 2.00 = NO`), no da descripción clínica. Nombre en transliteración, posible término persa relacionado con reanimación al nacer | `yes` / `NO` (casing mixto confirmado en el codebook fuente, igual que `IUGR`) | **Significado clínico sigue pendiente** — el codebook no tiene columna de descripción, solo nombre/nivel/valores; no asumir traducción sin validar con el equipo o la fuente original en Mendeley |

## Variables continuas

| Variable | Tipo | Descripción | Unidad | Rango (n=89) | Nulos |
|---|---|---|---|---|---|
| `PregnancyAge` | continua | Edad gestacional al nacer | semanas | 26.0 – 34.5 (media 32.1) | 0% |
| `BirthWeight` | continua | Peso al nacer | gramos | 670.5 – 2650.0 (media 1631.3) | 0% |
| `apgar1` | continua (discreta) | Puntaje Apgar al minuto 1 | escala 0–10 | 1 – 9 (media 7.1) | 0% |
| `apgar5` | continua (discreta) | Puntaje Apgar al minuto 5 | escala 0–10 | 3 – 10 (media 8.6) | 0% |
| `duration.hopitalization` | continua | Duración de hospitalización | días | 1 – 62 (media 19.0) | 0% |
| `duration.O2` | continua | Duración de oxígeno suplementario | días | 0 – 54 (media 6.3) | 0% |
| `RoundHeadAtBirth` | continua | Perímetro cefálico al nacer | cm (a confirmar unidad exacta) | 21.0 – 34.5 (media 29.6) | 0% |
| `number` | identificador | Número/ID de registro del neonato | entero | — | 0% — **no usar como feature**, es solo identificador |
| `Sex` | categórica binaria | Sexo del neonato | `male` / `female` | — | 0% |

## Variables con inconsistencia conocida (ver `FLAGGED_INCONSISTENT_VARS` en `loader.py`)

| Variable | Tipo | Descripción | Problema | Nulos |
|---|---|---|---|---|
| `correctedage` | ambigua | El codebook la tipa como `Scale` pero le asigna etiquetas ordinales: `1 = < -1 sd, 2 = < -2 sd, 3 = Normal` | **Contradicción confirmada en el codebook fuente**: los valores reales observados son enteros grandes (66 – 1190), consistentes con **edad en días**, no con una escala 1–3. No es un error de lectura del `.sav` — el codebook original ya es inconsistente con los datos que documenta | 4.5% (4/89) |
| `Age` | ambigua | Igual que `correctedage` (`1 = < -1 sd, 2 = < -2 sd, 3 = Normal` en el codebook) | Mismo problema — valores observados 130 – 1190 | 2.2% (2/89) |

**No usar estas dos columnas en modelado hasta que el equipo decida cómo tratarlas** (nota ya presente en el EDA de Bloque 3). Como el codebook fuente ya trae la inconsistencia, probablemente no se resuelve preguntándole al codebook — hay que decidir en equipo si se tratan como edad en días (hipótesis más consistente con los valores) o se descartan del modelado.

## Dominios de neurodesarrollo (Escalas Bayley)

| Variable | Tipo | Descripción |
|---|---|---|
| `CognitiveDomain`, `LanguageDomain`, `PerceptualDomain`, `finemotor`, `coarsemotor` | categórica (score crudo) | Resultado crudo de cada dominio de la Escala Bayley |
| `cog.recode`, `lang.recode`, `percep.recode`, `fine.recode`, `coarse.recode` | categórica | Recodificación intermedia de cada dominio (valores tipo `Normal`, `< -1 sd`, `< -2 sd`) |
| `cog.cat`, `lang.cat`, `percep.cat`, `fine.cat`, `coarse.cat` | categórica binaria | Versión final normal/abnormal de cada dominio — **insumo directo del target compuesto** |

Distribución individual por dominio (n=89, ver Bloque 3):

| Dominio | `normal` | `abnormal` | % abnormal |
|---|---|---|---|
| Cognitivo (`cog.cat`) | 62 | 27 | 30.3% |
| Lenguaje (`lang.cat`) | 61 | 28 | 31.5% |
| Perceptual (`percep.cat`) | 58 | 31 | 34.8% |
| Motor fino (`fine.cat`) | 67 | 22 | 24.7% |
| Motor grueso (`coarse.cat`) | 67 | 22 | 24.7% |

Acuerdo `fine.cat` vs `coarse.cat`: 91.0% (alta concordancia entre motricidad fina y gruesa).

## Variable objetivo

| Variable | Tipo | Descripción | Construcción |
|---|---|---|---|
| `neurodev_alteration` | categórica binaria (target) | Alteración del neurodesarrollo | `abnormal` si al menos uno de `cog.cat`, `lang.cat`, `percep.cat`, `fine.cat`, `coarse.cat` es `abnormal`; `normal` si todos son `normal`. Definida en `load_labeled()` (`src/data/loader.py`), no viene en el `.sav` original |

Distribución: 52.8% `abnormal` / 47.2% `normal` (ver README raíz, sección 1.1) — reemplazó a `sepsis` como target por el desbalance severo de esta última (5.6% positivos).

---

## Pendientes de este diccionario

- El codebook original (`data/raw/codebook.pdf`) solo documenta variable → nivel de medición → codificación de valores, **no tiene columna de descripción clínica**. Por eso `ehya.badve.tavallod`, `B.C`, `csf.culture` y `SepsisnegativeCulture` tienen su codificación confirmada pero su significado clínico exacto sigue siendo criterio de equipo (o consultar el artículo asociado en Mendeley/ScienceDirect).
- Re-verificar rangos/nulos/categorías contra las 89 filas completas corriendo el notebook con los datos ya en `data/raw/` (esta tabla se armó originalmente con outputs ya guardados del EDA de Bloque 3, sobre una muestra de 5 filas visible más agregados de `describe()`/`isnull()`; ya se pueden re-verificar directamente).
- Confirmar unidad exacta de `RoundHeadAtBirth` (asumido cm, el codebook no lo aclara).
- Decidir tratamiento de `type.of.ressucitation` (66% nulos) y de `correctedage`/`Age` (contradicción confirmada entre el codebook fuente y los valores observados) antes de la fase de features.
