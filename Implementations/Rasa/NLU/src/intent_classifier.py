#!/usr/bin/env python

'''
Intent Classification and their analysis
'''

import numpy as np
from tensorflow.keras import models, layers, callbacks
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import data_generation

def model_compile_train(model,
                        x_padded,
                        y_padded,
                        validation_data=None,
                        comp_loss="sparse_categorical_crossentropy",
                        comp_metrics="accuracy",
                        comp_optimizer="adam", 
                        callbacks=None, 
                        epochs=50,
                        validation_split=0.0,
                        batch_size=None,
                        shuffle=True,
                        save=False, 
                        title='model'
                        ):
    '''
    Used for merging the model.compile and model.fit for cleaner and easy invokation.
    Returns : model and history of training.
    '''
    model.compile(loss=comp_loss, metrics=comp_metrics, optimizer=comp_optimizer)
    history = model.fit(
            x_padded,
            y_padded,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_split=validation_split,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    if save:
        models.save_model(model, "../models/"+title+".h5")
    return model, history

def plot_train_val_analysis(history, title):
    
    '''Will plot the training and validation accuracy and loss graph/curves'''
    epoch_range = range(len(history.history["loss"]))
    # validation loss and training loss
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.title(title + " Loss graph")
    plt.plot(epoch_range, history.history["loss"], label="Training Loss")
    plt.plot(epoch_range, history.history["val_loss"], label="Validation Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title(title + " Accuracy graph")
    plt.plot(epoch_range, history.history["accuracy"], label="Training Accuracy")
    plt.plot(epoch_range, history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("../plots/" + title + "_loss_acc_graph")

def predict_max(model, x_test_padded):
    '''Predict the max argument of the all classes probability distribution'''
    y_pred = model.predict(x_test_padded)
    y_pred = [np.argmax(x) for x in y_pred]
    return y_pred

def generate_conf_matrix(y_test, y_pred, save=True, title='model'):
    '''Generate the confusion matrix of the classes'''
    disp = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(disp, annot=True, cmap="Blues").set(
        title="Confusion Matrix of " + title + "model"
    )
    if save:
        plt.savefig("../plots/" + title + "_confusion_matrix")
def generate_class_histgram(labels):
    label_count = labels.value_counts()
    sns.barplot(label_count.index, label_count)
    plt.xticks(rotation=20)
    plt.gca().set_ylabel("no of samples")
    plt.gca().set_xlabel('classes')

if __name__ == '__main__':
    df = data_generation.import_dataframe("../data/final_data.csv")
    generate_class_histgram(df['Intent'])