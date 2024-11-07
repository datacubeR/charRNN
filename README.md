# CharRNN

Repo con una demo de una CLI simple para un modelo generador de texto utilizando Redes LSTM.

## Instrucciones de uso

Para entrenar el modelo ejecutar:

```python
python -m charRNN.train -e 30 -w 100
```
donde los flags `-e` y `-w` corresponden al número de épocas y el largo de secuencia respectivamente.

Para generar texto utilizar el siguiente comando:

```python
python -m charRNN -cp CharRNN_30 -t 0.3 -c 1000 -p
```

donde `-cp CharRNN_30` es el nombre de un Checkpoint entrenado a 30 epochs (se pueden utilizar los generados por el proceso de entrenamiento), `-t` es la temperatura y `-c` es el número de caractéres a generar. El flag `-p` es para generar texto de manera probabilística (que suele dar mejores resultados). En caso de no usarlo, se genera de manera determinística.