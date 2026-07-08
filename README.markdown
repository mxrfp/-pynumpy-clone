# NumPy Clone in Puro Python (numpy_copy)

Una libreria leggera e scritta interamente in Python puro per la manipolazione di array N-dimensionali, fortemente ispirata a NumPy. 

Questo progetto è stato sviluppato come progetto personale e didattico durante il secondo anno di scuole superiori, con l'obiettivo di comprendere a fondo la Programmazione Orientata agli Oggetti (OOP), la ricorsione, il broadcasting vettoriale e il funzionamento dei metodi magici (dunder methods) di Python.

## Caratteristiche Principali

* Array N-Dimensionali: Gestione di matrici a qualsiasi dimensione tramite liste annidate e ricorsione.
* Operazioni Vettorializzate: Supporto per addizione, sottrazione, moltiplicazione, divisione, divisione intera e potenze (sia tra array che tra array e scalari).
* Tipi di Dato (Dtypes): Supporto integrato per interi (int), decimali (float) e numeri complessi (complex).
* Integrazione con Matplotlib: Implementazione nativa del protocollo `__array__`, che permette a librerie esterne come Matplotlib di leggere e renderizzare direttamente gli oggetti della libreria.
* Funzioni di Utilita: Implementazione di funzioni di generazione come `zeros`, `linspace`, `arange` e `meshgrid`.
* Modulo Random: Generazione di array casuali N-dimensionali tramite il modulo separato `random_nc`.

## Come utilizzarlo

Essendo scritto in puro Python, non richiede installazioni tramite pip (fatta eccezione per librerie opzionali esterne come Matplotlib per i grafici) e non utilizza moduli compilati in C. E' sufficiente clonare la repository e importare il modulo nel proprio script.

```python
import numpy_copy as nc

# Creazione di array e operazioni di base
a = nc.Array([[1, 2], [3, 4]])
b = nc.Array([[10, 20], [30, 40]])

# Broadcasting e vettorizzazione
risultato = (a + b) * 2
print(risultato)
```

## Esempio Avanzato: Frattale di Mandelbrot

Grazie al supporto per i numeri complessi e alla possibilita di eseguire calcoli su intere griglie di dati contemporaneamente, la libreria e in grado di calcolare insiemi matematici avanzati con poche righe di codice.

```python
import numpy_copy as nc
import matplotlib.pyplot as plt

# Creazione del piano complesso
punti = 150 
x = nc.linspace(-2, 2, punti, dtype=complex)
y = nc.linspace(-2, 2, punti, dtype=complex) * 1j

grid = nc.meshgrid(x, y)
c_plane = grid[0] + grid[1]

# Inizializzazione della matrice Z
z = nc.zeros(c_plane.shape, dtype=complex)

# Iterazione del frattale
for i in range(20):
    z = z * z + c_plane

# Visualizzazione
plt.imshow(abs(z), cmap='hot', extent=[-2, 2, -2, 2])
plt.title("Mandelbrot generato in puro Python")
plt.show()
```

## Limiti Noti e Possibili Sviluppi Futuri

Trattandosi di uno strumento puramente didattico che fa affidamento su liste annidate di Python e sulla ricorsione, le performance matematiche non sono equiparabili a quelle della libreria ufficiale NumPy.

Lavorare con matrici di dimensioni gigantesche puo comportare un rallentamento dei calcoli o il raggiungimento del limite di ricorsione nativo di Python (sebbene sia stato temporaneamente mitigato tramite `sys.setrecursionlimit`).
