# RememberUs

El Alzheimer es un tipo de enfermedad mental neurodegenerativa que como primer paso afecta la memoria y el pensamiento lógico. El objetivo principal de este proyecto es entrenar diferentes modelos de IA para predecir el desarrollo de Alzheimer en una persona. Ademas combinamos los modelos predictivos con un chatbot basdo en el el LLM [DeepSeek R1 Zero](https://openrouter.ai/deepseek/deepseek-r1-zero:free) para que cualquier persona pueda probar de primera mano el funcionamineto de los modelos de prediccion. [Prueba Aquí!!!](https://huggingface.co/[placeholder])

## Tabla de contenidos

1. [Arquitectura](#Arquitectura)
2. [Proceso](#Proceso)
3. [Funcionalidades](#Funcionalidades)
4. [Estado del proyecto](#EstadoDelProyecto)
5. [Agradecimientos](#Agradecimientos)


## Arquitectura 

Tanto el codigo de el procesamiento de datos y la creación de los modelos predictivos, está montado en un kernel Python en un Jupiter Notebook, haciendo uso de librerias como:

- Numpy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- Keras 
- ?

Y para el acceso remoto a este proyecto, utilizamos este repositorio en GitHub

## Proceso de desarrollo

Empezamos con el procesamiento de un dataset con datos de personas que padecen Alzheimer y otras que no, obtenido de [Fuente del dataset](https://www.kaggle.com/datasets/ankushpanday1/alzheimers-prediction-dataset-global).

Para empezar el dataset original cuenta con los siguientes tipos de columnas

![Proceso](assets/columntype.png)
