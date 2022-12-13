import tkinter as tkr
from tkinter import Tk
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from collections import OrderedDict

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix

dataFile = None
ErrorrateMeans = list()
AccuracyMeans = list()


def browse():
    Tk().withdraw()
    global dataFile
    dataFile = askopenfilename()
    print(dataFile)


def user_input_window():
    user_window = tkr.Toplevel(root)
    user_window.geometry("800x900")
    user_window.resizable(0, 0)

    def predicted_label(label):
        if label == 1:
            account_type_label = tkr.Label(user_window, text="Account Type : Fake").grid(row=24)
        else:
            account_type_label = tkr.Label(user_window, text="Account Type : Not Fake").grid(row=24)

    def manualInputNaiveBayes():
        if dataFile:
            print("here")

            if e1:
                inputframe = pd.DataFrame(
                    OrderedDict(
                        {
                            'ProfileID': [e1.get()],
                            'Count Of Abuses Reported': [e2.get()],
                            'Count Of Friend Requests Rejected': [e3.get()],
                            'Count Of UnAccepted Friend Requests ': [e4.get()],
                            'Count Of Friends': [e5.get()],
                            'No Of Followers': [e6.get()],
                            'Count Of Likes To Unknown Account': [e7.get()],
                            'Count Of Comments Per Day': [e8.get()],
                        }))
            inputframe = inputframe[
                ['ProfileID', 'Count Of Abuses Reported', 'Count Of UnAccepted Friend Requests ', 'Count Of Friends',
                 'No Of Followers', 'Count Of Likes To Unknown Account', 'Count Of Comments Per Day']]
            print(inputframe.loc[0])

            fetchData = pd.read_csv(dataFile)
            mask = np.random.rand(len(fetchData)) < 0.7
            training_data = fetchData[mask]
            test_set = inputframe
            testing_data = test_set.values[:, 0:7]
            features = training_data.values[:, 0:7]
            labels = training_data.values[:, 8].astype('int')
            model1 = MultinomialNB()
            model1.fit(features, labels)
            predictions_model1 = model1.predict(testing_data)

            print('\n1.Naive Bayes Prediction:\n')
            print('\n Predicted Class :', predictions_model1[0])
            predicted_label(predictions_model1[0])

    def ManualInput_LinearSVC():
        if dataFile:
            print("here")
            if e1:
                inputframe = pd.DataFrame(
                    OrderedDict(
                        {
                            'ProfileID': [e1.get()],
                            'Count Of Abuses Reported': [e2.get()],
                            'Count Of Friend Requests Rejected': [e3.get()],
                            'Count Of UnAccepted Friend Requests ': [e4.get()],
                            'Count Of Friends': [e5.get()],
                            'No Of Followers': [e6.get()],
                            'Count Of Likes To Unknown Account': [e7.get()],
                            'Count Of Comments Per Day': [e8.get()],
                        }))
            inputframe = inputframe[
                ['ProfileID', 'Count Of Abuses Reported', 'Count Of UnAccepted Friend Requests ', 'Count Of Friends',
                 'No Of Followers', 'Count Of Likes To Unknown Account', 'Count Of Comments Per Day']]
            print(inputframe.loc[0])

            fetchData = pd.read_csv(dataFile)
            mask = np.random.rand(len(fetchData)) < 0.7
            training_data = fetchData[mask]
            test_set = inputframe
            testing_data = test_set.values[:, 0:7]
            features = training_data.values[:, 0:7]
            labels = training_data.values[:, 8].astype('int')
            model2 = LinearSVC()
            model2.fit(features, labels)
            predictions_model2 = model2.predict(testing_data)

            print('2.Linear SVC Prediction:\n')
            print('\n Predicted Class :', predictions_model2[0])
            predicted_label(predictions_model2[0])

    def ManualInput_KNN():
        if dataFile:
            print("here")

            if e1:
                inputframe = pd.DataFrame(
                    OrderedDict(
                        {
                            'ProfileID': [e1.get()],
                            'Count Of Abuses Reported': [e2.get()],
                            'Count Of Friend Requests Rejected': [e3.get()],
                            'Count Of UnAccepted Friend Requests ': [e4.get()],
                            'Count Of Friends': [e5.get()],
                            'No Of Followers': [e6.get()],
                            'Count Of Likes To Unknown Account': [e7.get()],
                            'Count Of Comments Per Day': [e8.get()],
                        }))
            inputframe = inputframe[
                ['ProfileID', 'Count Of Abuses Reported', 'Count Of UnAccepted Friend Requests ', 'Count Of Friends',
                 'No Of Followers', 'Count Of Likes To Unknown Account', 'Count Of Comments Per Day']]
            print(inputframe.loc[0])

            fetchData = pd.read_csv(dataFile)
            mask = np.random.rand(len(fetchData)) < 0.7
            training_data = fetchData[mask]
            test_set = inputframe
            testing_data = test_set.values[:, 0:7]
            features = training_data.values[:, 0:7]
            labels = training_data.values[:, 8].astype('int')
            model3 = KNeighborsClassifier(n_neighbors=3)
            model3.fit(features, labels)
            predictions_model3 = model3.predict(testing_data)
            print('3.KNN Prediction :\n')
            print('\n Predicted Class :', predictions_model3[0])
            predicted_label(predictions_model3[0])

    tkr.Label(user_window, text="ProfileID").grid(row=0)
    tkr.Label(user_window, text="Count Of Abuses Reported").grid(row=3)
    tkr.Label(user_window, text="No Of Count Of Friend Requests Rejected").grid(row=5)
    tkr.Label(user_window, text="Count Of UnAccepted Friend Requests ").grid(row=7)
    tkr.Label(user_window, text="Count Of Friends").grid(row=9)
    tkr.Label(user_window, text="No Of Followers").grid(row=11)
    tkr.Label(user_window, text="Count Of Likes To Unknown Account").grid(row=13)
    tkr.Label(user_window, text="Count Of Comments Per Day").grid(row=15)

    e1 = tkr.Entry(user_window)
    e2 = tkr.Entry(user_window)
    e3 = tkr.Entry(user_window)
    e4 = tkr.Entry(user_window)
    e4 = tkr.Entry(user_window)
    e5 = tkr.Entry(user_window)
    e6 = tkr.Entry(user_window)
    e7 = tkr.Entry(user_window)
    e8 = tkr.Entry(user_window)

    e1.grid(row=2, column=0)
    e2.grid(row=4, column=0)
    e3.grid(row=6, column=0)
    e4.grid(row=8, column=0)
    e5.grid(row=10, column=0)
    e6.grid(row=12, column=0)
    e7.grid(row=14, column=0)
    e8.grid(row=16, column=0)

    tkr.Label(user_window,
              text="Compute Prediction",
              fg="gold",
              bg="black",
              width=45,
              height=2,
              font="Calibri 18 bold").grid(row=32)

    button_NaiveBayes = tkr.Button(user_window,
                                    text="Naive Bayes",
                                    fg="gold",
                                    bg="black",
                                    width=25,
                                    height=2,
                                    command=manualInputNaiveBayes,
                                    font="Calibri 10 bold",
                                    )
    button_NaiveBayes.place(relx=0.30, rely=0.7)

    button_LinearSVC = tkr.Button(user_window,
                                   text="Linear SVC",
                                   fg="gold",
                                   bg="black",
                                   width=25,
                                   height=2,
                                   font="Calibri 10 bold",
                                   command=ManualInput_LinearSVC,
                                   )
    button_LinearSVC.place(relx=0.30, rely=0.6)

    button_KNN = tkr.Button(user_window,
                            text="KNN",
                            fg="gold",
                            bg="black",
                            width=25,
                            height=2,
                            font="Calibri 10 bold",
                            command=ManualInput_KNN,
                            )
    button_KNN.place(relx=0.30, rely=0.5)


def Naive_Bayes():
    if dataFile:
        global AccuracyMeans, ErrorrateMeans
        fetchData = pd.read_csv(dataFile)
        mask = np.random.rand(len(fetchData)) < 0.7
        training_data = fetchData[mask]
        test_set = fetchData[~mask]
        testing_data = test_set.values[:, 0:7]
        testing_data_labels = test_set.values[:, 8]
        features = training_data.values[:, 0:7]
        labels = training_data.values[:, 8].astype('int')
        model1 = MultinomialNB()
        model1.fit(features, labels)
        predictions_model1 = model1.predict(testing_data)

        accuracy = accuracy_score(testing_data_labels, predictions_model1) * 100
        AccuracyMeans.append(accuracy)
        error_rate = 100 - accuracy
        ErrorrateMeans.append(error_rate)
        precision = precision_score(testing_data_labels, predictions_model1) * 100
        recall = recall_score(testing_data_labels, predictions_model1) * 100

        print('\n1.Naive Bayes Prediction:\n')
        print('Confusion Matrix :')
        print(confusion_matrix(testing_data_labels, predictions_model1))
        print('Accuracy Is : ' + str(accuracy) + ' %')
        print('Error Rate Is : ' + str(error_rate) + ' %')
        print('Precision Is : ' + str(precision) + ' %')
        print('Recall Is : ' + str(recall) + ' %\n\n')

        labels = ['Error Rate', 'Accuracy ']
        sizes = [error_rate, accuracy]
        explode = (0, 0.1)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.title('Naive Bayes Algorithm')
        ax1.axis('equal')
        plt.tight_layout()
        plt.show()


def Linear_Svc():
    if dataFile:
        global AccuracyMeans, ErrorrateMeans
        fetchData = pd.read_csv(dataFile)
        mask = np.random.rand(len(fetchData)) < 0.7
        training_data = fetchData[mask]
        test_set = fetchData[~mask]
        testing_data = test_set.values[:, 0:7]
        testing_data_labels = test_set.values[:, 8]
        features = training_data.values[:, 0:7]
        labels = training_data.values[:, 8].astype('int')
        model2 = LinearSVC()
        model2.fit(features, labels)
        predictions_model2 = model2.predict(testing_data)

        accuracy = accuracy_score(testing_data_labels, predictions_model2) * 100
        AccuracyMeans.append(accuracy)
        error_rate = 100 - accuracy
        ErrorrateMeans.append(error_rate)
        precision = precision_score(testing_data_labels, predictions_model2) * 100
        recall = recall_score(testing_data_labels, predictions_model2) * 100

        print('2.Linear SVC Prediction:\n')
        print('Confusion Matrix :')
        print(confusion_matrix(testing_data_labels, predictions_model2))
        print('Accuracy Is : ' + str(accuracy) + ' %')
        print('Error Rate Is : ' + str(error_rate) + ' %')
        print('Precision Is : ' + str(precision) + ' %')
        print('Recall Is : ' + str(recall) + ' %\n\n')

        labels = ['Error Rate', 'Accuracy ']
        sizes = [error_rate, accuracy]
        explode = (0, 0.1)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)

        plt.title('Linear SVC Algorithm')
        ax1.axis('equal')
        plt.tight_layout()
        plt.show()


def Knn():
    if dataFile:
        global AccuracyMeans, ErrorrateMeans
        fetchData = pd.read_csv(dataFile)
        mask = np.random.rand(len(fetchData)) < 0.7
        training_data = fetchData[mask]
        test_set = fetchData[~mask]
        testing_data = test_set.values[:, 0:7]
        testing_data_labels = test_set.values[:, 8]
        features = training_data.values[:, 0:7]
        labels = training_data.values[:, 8].astype('int')

        model3 = KNeighborsClassifier(n_neighbors=3)
        model3.fit(features, labels)
        predictions_model3 = model3.predict(testing_data)

        accuracy = accuracy_score(testing_data_labels, predictions_model3) * 100
        AccuracyMeans.append(accuracy)
        error_rate = 100 - accuracy
        ErrorrateMeans.append(error_rate)
        precision = precision_score(testing_data_labels, predictions_model3) * 100
        recall = recall_score(testing_data_labels, predictions_model3) * 100

        print('3.KNN Prediction  :\n')
        print('Confusion Matrix :')
        print(confusion_matrix(testing_data_labels, predictions_model3))
        print('Accuracy Is : ' + str(accuracy) + ' %')
        print('Error Rate Is : ' + str(error_rate) + ' %')
        print('Precision Is : ' + str(precision) + ' %')
        print('Recall Is : ' + str(recall) + ' %\n\n')

        labels = ['Error Rate', 'Accuracy ']
        sizes = [error_rate, accuracy]

        explode = (0, 0.1)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)

        plt.title('KNN Algorithm')
        ax1.axis('equal')
        plt.tight_layout()
        plt.show()


def compare():
    N = 3
    ind = np.arange(N)
    width = 0.45
    p1 = plt.bar(ind, AccuracyMeans, width)
    p2 = plt.bar(ind, ErrorrateMeans, width, bottom=AccuracyMeans)
    plt.ylabel('Scores')
    plt.title('Classifiers Performance')
    plt.xticks(ind, ('Naive Bayes', 'Linear SVC', 'KNN',))
    plt.yticks(np.arange(0, 120, 20))
    plt.legend((p1[0], p2[0]), ('Accuracy', 'Error Rate'))
    plt.show()


root = tkr.Tk()
root.title("Fake Account Detector for Instagram")
root.grid_columnconfigure(0, weight=1)
root.geometry("600x650")
root.resizable(0, 0)

tkr.Label(root,
          text="Fake Account Detector for Instagram",
          fg="dark blue",
          bg="light blue",
          width=400,
          height=2,
          font="Calibri 25 bold italic").pack()

browsebutton = tkr.Button(root,
                          text="Select Dataset",
                          fg="black",
                          bg="white",
                          width=20,
                          height=2,
                          font="Calibri 12 bold italic",
                          command=browse,
                          )
browsebutton.pack(padx=0, pady=10)

tkr.Label(root,
          text="Classifiers ",
          fg="dark blue",
          bg="light blue",
          width=40,
          height=1,
          font="Calibri 25 bold italic").pack(padx=50, pady=5)

Manual_Input_Button = tkr.Button(root,
                                 text="Give Manual Input",
                                 fg="gold",
                                 bg="black",
                                 width=35,
                                 height=2,
                                 font="Calibri 15 bold italic",
                                 command=user_input_window,
                                 )
Manual_Input_Button.place(relx=0.20, rely=0.40)

button_statistics = tkr.Button(root,
                                   text="Compare",
                                   fg="black",
                                   bg="light steel blue",
                                   width=35,
                                   height=2,
                                   font="Calibri 12 bold italic",
                                   command=compare,
                                   )
button_statistics.place(relx=0.25, rely=0.9)

button_NaiveBayes = tkr.Button(root,
                                text="Naive Bayes",
                                fg="gold",
                                bg="black",
                                width=20,
                                height=2,
                                command=Naive_Bayes,
                                font="Calibri 15 bold italic",
                                )
button_NaiveBayes.pack(side=tkr.LEFT)

button_LinearSVC = tkr.Button(root,
                               text="Linear SVC",
                               fg="gold",
                               bg="black",
                               width=20,
                               height=2,
                               font="Calibri 15 bold italic",
                               command=Linear_Svc,
                               )
button_LinearSVC.pack(side=tkr.LEFT)

button_KNN = tkr.Button(root,
                        text="KNN",
                        fg="gold",
                        bg="black",
                        width=20,
                        height=2,
                        font="Calibri 15 bold italic",
                        command=Knn,
                        )
button_KNN.pack(side=tkr.LEFT)
root.mainloop()