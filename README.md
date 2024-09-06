# Per vedere la Web Page [Progetto WoT - Palumbo & Pierri](https://unisalento-idalab-iotcourse-2023-2024.github.io/wot-project-presentation-PalumboPierri/ )
# Classificazione Audio con Machine Learning

Questo progetto dimostra un sistema di classificazione audio utilizzando una rete neurale addestrata sul dataset UrbanSound8K. Il modello è in grado di classificare suoni ambientali come condizionatori, sirene, trapani, ecc.

## Funzionalità

- **Estrazione di Caratteristiche Audio**: Il sistema estrae diverse caratteristiche audio, come MFCC, cromatiche, contrasto spettrale e altre dai file audio.
- **Data Augmentation**: Vengono applicate tecniche di aumento dei dati come pitch shift, time stretching e aggiunta di rumore per migliorare la generalizzazione del modello.
- **Rete Neurale**: Una rete neurale completamente connessa costruita con TensorFlow/Keras per classificare i segnali audio.
- **Predizione**: Il modello può essere utilizzato per prevedere la classe di un nuovo file audio.

## Dataset

Il progetto utilizza il dataset [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html), che contiene circa 8.000 file audio classificati in 10 diverse categorie di suoni urbani.

## Requisiti

Per eseguire il progetto, sono necessari i seguenti pacchetti Python:

- `numpy`
- `pandas`
- `librosa`
- `soundfile`
- `tensorflow`
- `scikit-learn`

Puoi installare tutti i requisiti utilizzando il seguente comando:

```bash
pip install numpy pandas librosa soundfile tensorflow scikit-learn
```
## Utilizzo
- Preparazione del Dataset: Scarica e posiziona il dataset UrbanSound8K nella cartella appropriata.
- Esecuzione del Codice: Esegui lo script Python per addestrare il modello e classificare i suoni.
```bash
python machine.py
```
## Risultati
Il modello raggiunge una precisione di circa il 70% sulla classificazione dei suoni utilizzando le caratteristiche audio estratte e le tecniche di data augmentation.


