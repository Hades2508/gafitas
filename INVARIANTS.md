# LOCAL_PROGRAMMER_V0 — invariantes del runner

Pieza **permanente**. No es un arnés de campaña. Se reutiliza para modelos
locales, Luna, modelos futuros, single-file y (más adelante) multi-file, sin
reescribir infraestructura.

Contrato congelado que implementa:
`D:\GATE-A-WS\LOCAL_PROGRAMMER_V0\LOCAL_PROGRAMMER_V0_CONTRACT.md`

Implementado por Claude bajo **excepción explícita del operador**
(`generalista.md` excepción 4, con componente de excepción 3: implementación de
confianza de seguridad e infraestructura). El registro lo exige la regla.

---

## I1 — Cuatro clases de fallo, cuatro destinos. No se mezclan.

| clase | destino | nunca |
|---|---|---|
| `ToolError` | texto de feedback al modelo; **el bucle sigue** | no cuenta como llamada inválida |
| `InvalidCall` | cuenta en `llamadas_invalidas` **y** vuelve al modelo | no aborta |
| `ProviderError` | resultado estructurado y persistido; `scoreable=False` | **jamás se puntúa como fallo del modelo** |
| `HarnessInvalid` | anula la run, preserva traza y workspace | lo produce **solo** un defecto de este repo |

`tools.dispatch` convierte **cualquier** excepción que no sea de las tres
primeras en `HarnessInvalid`. Una herramienta que olvide contemplar un caso
detiene la medición y se delata, en vez de corromper una métrica en silencio.

## I2 — Un solo bucle.

`loop.run_loop` es el único. Screen y Step 1 lo comparten. Dos bucles ya
divergieron una vez (`run` vs `run_tests`, un `list_dir` fantasma) antes de
que ninguno fuera de fiar.

## I3 — Un solo origen para el esquema de herramientas.

`tools.SPECS` genera el esquema nativo **y** valida los argumentos. No pueden
divergir. Un `assert` a nivel de módulo exige que `SPECS` y las
implementaciones describan exactamente el mismo conjunto.

## I4 — Siete herramientas. Ni una más.

`read_file · grep · list_symbols · edit · write_file · run_tests · finish`

Cualquier otro nombre es `ERROR_UNKNOWN_TOOL` (llamada inválida), nunca un
fallo. No hay shell libre. `argv` de pytest lo construye el arnés.

## I5 — Contención prestada, nunca reimplementada.

`programmer.guard` en **toda** escritura y **toda** resolución de ruta.
`deps.py` fija la raíz, verifica el `__file__` del módulo importado y lanza
`HarnessInvalid` si algo lo ha ensombrecido. Sin guard **no se ejecuta**: una
run sin contención no es una medición más débil, no es una medición.

## I6 — Lectura amplia, escritura estrecha.

Se lee todo el worktree. Se escribe solo en `write_scope` (+
`allowed_new_files` al crear). La jaula de lectura fue el error caro; la de
escritura no cuesta capacidad.

## I7 — La caja es el rollback.

Worktree git cuando el repo lo permite, copia aislada cuando no —y queda
**registrado cuál**. El repo de origen nunca se toca.

## I8 — Se conserva lo que falla.

`dispose(preserve=...)`: se borra en éxito, **se conserva en fallo**. El
runner anterior borraba en un `finally` y cada fallo destruía su propia
evidencia, incluidos los fallos del propio arnés.

## I9 — El veredicto lo toma el arnés, después del bucle.

Nunca la afirmación del modelo, nunca una lectura a mitad de run. `finish`
solo cede el turno.

## I10 — Presupuesto duro.

12 turnos, `num_ctx` 8192, `num_predict` 1024. Agotarlo es un **resultado**
(`BUDGET_EXHAUSTED`), no un error.

## I11 — Elisión, no presupuesto de contexto.

Resultados de herramienta de más de 3 turnos y >500 caracteres se sustituyen
por un marcador. **Nunca** se elide: la llamada, ningún error, ni los 3
últimos turnos. Sustituye a las 482 líneas de `source_context.py`.

## I12 — La telemetría nunca rompe una run.

`DuplicateMissionError` se registra y se salta. Un store inaccesible degrada a
`NullTelemetry` y se anota.

## I13 — Prompts y puntuación congelados.

`screen.SYSTEM_PROMPT` es §D.3 byte a byte. La puntuación (§D.5) y los
umbrales (§E) se leen, no se ajustan. **Ninguna** variación por modelo.

## I14 — Una desviación, declarada.

`screen.PROTOCOL_B_SUFFIX` añade el formato de llamada de §D.4 solo en
protocolo B, idéntico para todos los modelos, y se registra como
`protocol_b_suffix_applied` en cada resultado. §D.3 queda intacto.

**Motivo:** §D.3 nunca comunicaba la sintaxis `LLAMADA:`. Un protocolo cuya
sintaxis no se enuncia mide adivinación, no al modelo. **El operador puede
vetarlo borrando la constante.**

---

## Defectos previos cubiertos por la suite

| defecto | test |
|---|---|
| tool schema incompleto (`list_dir` anunciado sin implementar, `run` vs `run_tests`) | `test_native_schema_covers_every_tool_and_nothing_else`, `test_schema_required_matches_specs` |
| provider HTTP 500 / timeout / body inválido / transporte | `test_provider_failure_is_structured_and_not_scoreable` (×4), `test_provider_failure_on_the_first_turn_still_records` |
| syntax error propagado que tumbaba la campaña | `test_list_symbols_on_broken_file_is_a_tool_error`, `test_edit_that_breaks_syntax_writes_nothing` |
| `list_dir` sobre fichero / ruta del tipo equivocado | `test_unknown_tool_is_an_invalid_call_not_a_crash`, `test_read_file_on_directory_is_a_tool_error`, `test_list_symbols_on_directory_is_a_tool_error` |
| argumentos ausentes / mal formados / `arguments` como string | `test_missing_required_argument_is_an_invalid_call`, `test_arguments_as_json_string_are_accepted`, `test_arguments_that_are_a_list_are_an_invalid_call` |
| tool errors que tumbaban el runner | `test_a_tool_error_feeds_back_and_the_loop_continues`, `test_the_error_text_actually_reaches_the_model` |
| regex de protocolo B que corrompía `=` dentro de strings | `test_protocol_b_preserves_equals_inside_strings` y 7 más |
| `grep`/`write_file` que eran stubs y no hacían nada | `test_grep_finds_matches`, `test_write_file_creates_an_allowed_new_file` |
| `SAME_FILE_MULTI_OP_COLLISION` (vivo en Generalista) | `test_two_edits_to_one_file_both_survive` |
| misión malformada puntuada como fallo del modelo | `test_malformed_mission_is_harness_invalid_not_a_model_failure` |
| `DuplicateMissionError` que mataba un re-run | `test_telemetry_duplicate_is_recorded_not_raised` |
| workspace borrado en `finally`, evidencia destruida | `test_workspace_is_preserved_on_failure_and_removed_on_success`, `test_screen_provider_failure_does_not_score_the_model` |

---

## Uso

```
python -m localprog selfcheck
python -m localprog screen --out <DIR> --models <M> [--runs 3] [--protocol-a-only]
python -m localprog step1 --out <DIR> --model <M> --missions <a.json> <b.json> [--telemetry-db <db>]
```

Ningún subcomando hace nada por defecto: todos exigen `--out` explícito.
Construir el arnés y ejecutar el gate son decisiones distintas.

## Pendiente del operador — dos cosas, ninguna tocada por mí

1. **`screen_runner.py` y `step1_runner.py` están commiteados dentro de
   `D:\PROGRAMMER-BUILD-ROOT`** (Gafotas, autoridad congelada), commits
   `cf88154`, `23c828c`, `e1760a4`. Repositorio equivocado, y exactamente la
   clase de duplicado que ya causó un incidente de shadowing. No los he
   borrado: es otra autoridad y no tengo autorización sobre ella.
2. **`D:\GATE-A-WS\LOCAL_PROGRAMMER_V0\SCREEN_V0\`** conserva su
   `CONTAMINATION_NOTICE.md` y es correcto: `qwen2.5-coder:7b` marcó 0/5 con
   12/12 llamadas inválidas en las 6 runs, que es una firma de arnés, no de
   modelo. Esa evidencia se preserva y se excluye; el re-run va con id nuevo.

## I15 — El motor se usa virgen. Nunca se entrena, ajusta ni adapta.

GAFITAS mejora **el cuerpo**, jamás el cerebro. Un motor entra tal y como lo
publicó quien lo hizo: sin fine-tuning, sin LoRA, sin adaptadores, sin
destilación, sin prompts grabados en pesos, sin cuantización propia hecha para
que un número suba.

Prohibido, no desaconsejado:

| prohibido | por qué |
|---|---|
| fine-tuning / SFT | el resultado deja de transferir a otro motor |
| LoRA / adaptadores | lo mismo, con menos ficheros |
| destilación a un alumno propio | eso fabrica un motor, no lo soporta |
| cuantización a medida para subir una métrica | mide el cuantizador, no el sistema |
| cualquier peso tocado por nosotros | deja de ser el motor que dice ser |

**Por qué es la distinción y no una limitación.** Una mejora dentro de los pesos
sirve a un motor. Una mejora dentro del arnés sirve a todos los que entren
después, incluidos los que aún no existen. Es lo que hace que "multimotor"
signifique algo comprobable en vez de una lista de modelos compatibles.

**El contraste está medido, y no es nuestro.** TinyAgent (arXiv 2409.00608,
Berkeley) llevó un 1.1B del 12.71% al 80% en function calling: 80.000 muestras
sintéticas, ~500 dólares de GPT-4-Turbo generándolas, y fine-tuning. El
resultado es excelente y pertenece a ese checkpoint. En este repo,
granite4.1:3b fue del 4% al 60% en un día con el modelo intacto, cero llamadas
de pago y cero pesos tocados -- y todo lo que lo consiguió (repair, legal_tools,
copy_code, las correcciones que nombran el trabajo, el cierre limpio) está
disponible para cualquier motor que se conecte mañana sin repetir nada.

Corolario operativo: cuando un fallo parezca "el modelo no sabe", la pregunta no
es qué entrenarle. Es qué está haciendo el arnés que se lo impide. Todos los
defectos de esta campaña -- los nueve -- resultaron estar de este lado.
