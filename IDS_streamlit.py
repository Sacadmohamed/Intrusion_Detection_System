import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import tree
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc
from mlxtend.plotting import plot_confusion_matrix
import joblib

# Path to the CSV files
trained_data_path = r'C:\Users\Eng Sacad\Desktop\IDS\Trained_data.csv'
tested_data_path = r'C:\Users\Eng Sacad\Desktop\IDS\Tested_data.csv'

def load_data(file_path):
    # Load CSV file into a DataFrame
    df = pd.read_csv(file_path)
    return df

def visualize_box_plots(df):
    # Plot box plots for each column
    fig, ax = plt.subplots(figsize=(20, 40))
    df.plot(kind='box', subplots=True, layout=(8, 5), ax=ax)
    plt.show()
    st.pyplot(fig)

def DataProcessing():
    st.title('Data Processing')
    st.write('This is the content of the Data Processing tab')
    # Add your data processing code here

def Model_Evaluation():
    st.title('Model Evaluation')
    st.write('This is the content of the Model Evaluation tab')
    # Add your model evaluation code here


def load_csv_file():
    file_path = st.file_uploader("Upload CSV file", type=["csv"])
    if file_path is not None:
        Pred_Data = pd.read_csv(file_path)
        st.dataframe(Pred_Data)  # Display the DataFrame
        return Pred_Data  # Return the loaded data


# Streamlit web app
def main():
    # Set page title
    st.title("Intrusion Detection System")

    # Load trained data
    Trained_Data = load_data(trained_data_path)
    st.subheader("Uploading the Data for Training")
    st.write("Data preview:")
    st.dataframe(Trained_Data.head())
    st.write("Data shape:", Trained_Data.shape)

    # Load tested data
    Tested_Data = load_data(tested_data_path)
    st.subheader(" Uploading the Data forTesting")
    st.write("Data preview:")
    st.dataframe(Tested_Data.head())
    st.write("Data shape:", Tested_Data.shape)

    # Visualize box plots for trained data
    st.subheader("Box Plots for Trained Data")
    visualize_box_plots(Trained_Data)

    # Visualize box plots for tested data
    st.subheader("Box Plots for Tested Data")
    visualize_box_plots(Tested_Data)

    # Data preprocessing
    Trained_attack = Trained_Data['class'].map(lambda a: 0 if a == 'normal' else 1)
    Tested_attack = Tested_Data['class'].map(lambda a: 0 if a == 'normal' else 1)
    Trained_Data['attack_state'] = Trained_attack
    Tested_Data['attack_state'] = Tested_attack
    ## Data Encoding
    Trained_Data = pd.get_dummies(Trained_Data,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")
    ## Data Encoding
    Tested_Data = pd.get_dummies(Tested_Data,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")
    LE = LabelEncoder()
    attack_LE= LabelEncoder()
    Trained_Data['class'] = attack_LE.fit_transform(Trained_Data["class"])
    Tested_Data['class'] = attack_LE.fit_transform(Tested_Data["class"])

    ### Data Splitting
    X_train = Trained_Data.drop('class', axis = 1)
    X_train = Trained_Data.drop('attack_state', axis = 1)

    X_test = Tested_Data.drop('class', axis = 1)
    X_test = Tested_Data.drop('attack_state', axis = 1)


    Y_train = Trained_Data['attack_state']
    Y_test = Tested_Data['attack_state']

    X_train_train,X_test_train ,Y_train_train,Y_test_train = train_test_split(X_train, Y_train, test_size= 0.25 , random_state=42)
    X_train_test,X_test_test,Y_train_test,Y_test_test = train_test_split(X_test, Y_test, test_size= 0.25 , random_state=42)
    #### Data Scaling
    Ro_scaler = RobustScaler()
    X_train_train = Ro_scaler.fit_transform(X_train_train) 
    X_test_train= Ro_scaler.transform(X_test_train)
    X_train_test = Ro_scaler.fit_transform(X_train_test) 
    X_test_test= Ro_scaler.transform(X_test_test)

    X_train_train.shape, Y_train_train.shape
    X_test_train.shape, Y_test_train.shape
    X_train_test.shape, Y_train_test.shape
    X_test_test.shape, Y_test_test.shape


    def Evaluate(Model_Name, Model_Abb, X_test, Y_test):
    
      Pred_Value= Model_Abb.predict(X_test)
      Accuracy = metrics.accuracy_score(Y_test,Pred_Value)                      
      Sensitivity = metrics.recall_score(Y_test,Pred_Value)
      Precision = metrics.precision_score(Y_test,Pred_Value)
      F1_score = metrics.f1_score(Y_test,Pred_Value)
      Recall = metrics.recall_score(Y_test,Pred_Value)
      
      st.write('--------------------------------------------------\n')
      st.write('The {} Model Accuracy   = {}\n'.format(Model_Name, np.round(Accuracy, 3)))
      st.write('The {} Model Sensitivity = {}\n'.format(Model_Name, np.round(Sensitivity, 3)))
      st.write('The {} Model Precision  = {}\n'.format(Model_Name, np.round(Precision, 3)))
      st.write('The {} Model F1 Score   = {}\n'.format(Model_Name, np.round(F1_score, 3)))
      st.write('The {} Model Recall     = {}\n'.format(Model_Name, np.round(Recall, 3)))
      st.write('--------------------------------------------------\n')
    
      Confusion_Matrix = metrics.confusion_matrix(Y_test, Pred_Value)
      #plot_confusion_matrix(Confusion_Matrix,class_names=['Normal', 'Attack'],figsize=(5.55,5), colorbar= "blue")
      #plot_roc_curve(Model_Abb, X_test, Y_test)

      def GridSearch(Model_Abb, Parameters, X_train, Y_train):
       Grid = GridSearchCV(estimator=Model_Abb, param_grid= Parameters, cv = 3, n_jobs=-1)
       Grid_Result = Grid.fit(X_train, Y_train)
       Model_Name = Grid_Result.best_estimator_
    
      return (Model_Name)
    


    st.subheader("Logistic Regression:")
    ### Logistic Regression
    from sklearn.linear_model import LogisticRegression
    LR= LogisticRegression()
    LR.fit(X_train_train , Y_train_train)
    st.write("Score for the Logistic Regression:")
    LR.score(X_train_train, Y_train_train), LR.score(X_test_train, Y_test_train)

    Evaluate('Logistic Regression', LR, X_test_train, Y_test_train)

    ### Plottin the ROC
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    # Get predicted probabilities for positive class
    Y_prob = LR.predict_proba(X_test_train)[:, 1]

    # Compute false positive rate, true positive rate, and threshold
    fpr, tpr, thresholds = roc_curve(Y_test_train, Y_prob)

    # Compute the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    
    # Display the plot in Streamlit
    st.pyplot(plt)

  # Calculate confusion matrix
    y_pred = LR.predict(X_test_train)
    cm = confusion_matrix(Y_test_train, y_pred)

    # Plot the confusion matrix
    plt.figure()
    class_names = ['normal', 'attack']  # Specify class labels
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    ax.set_xticklabels([''] + class_names)
    ax.set_yticklabels([''] + class_names)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    # Display the confusion matrix chart in Streamlit
    st.pyplot(plt)


    st.subheader("Decision Tree:")
    #### Decision Tree
    DT =DecisionTreeClassifier(max_features=6, max_depth=4)
    DT.fit(X_train_train, Y_train_train)
    st.write("Score for the Desicison Tree:")
    DT.score(X_train_train, Y_train_train), DT.score(X_test_train, Y_test_train)
    
    Evaluate('Decision Tree Classifier', DT, X_test_train, Y_test_train)

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # Get predicted probabilities for positive class
    Y_prob = DT.predict_proba(X_test_train)[:, 1]

    # Compute false positive rate, true positive rate, and threshold
    fpr, tpr, thresholds = roc_curve(Y_test_train, Y_prob)
    # Compute the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    # Display the plot in Streamlit
    st.pyplot(plt)

  
    # Get predicted labels
    y_pred = DT.predict(X_test_train)
    # Calculate confusion matrix
    cm = confusion_matrix(Y_test_train, y_pred)
    # Plot the confusion matrix
    plt.figure()
    class_names = ['normal', 'attack']  # Specify class labels
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    ax.set_xticklabels([''] + class_names)
    ax.set_yticklabels([''] + class_names)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    # Display the confusion matrix chart in Streamlit
    st.pyplot(fig)
 
    st.write("Decision Tree Chart:")
    # Display the Decision Tree chart
    fig = plt.figure(figsize=(15,12))
    tree.plot_tree(DT, filled=True)
    st.pyplot(fig)




    st.subheader("Random Forest Classifier:")
    # Random Forest Classifier
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    parameters = {'max_depth': max_depth}

    RF = RandomForestClassifier()
    grid_search = GridSearchCV(RF, parameters)
    grid_search.fit(X_train_train, Y_train_train)

    # Retrieve the best model
    RF = grid_search.best_estimator_

    # Fit the model
    RF.fit(X_train_train, Y_train_train)
    
    st.write("Score for the Random Forest Classifier:")
    # Assuming you have defined the Evaluate function
    Evaluate('Random Forest Classifier', RF, X_test_train, Y_test_train)
    # Get predicted probabilities for positive class
    y_pred_prob = RF.predict_proba(X_test_train)[:, 1]

    # Compute false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(Y_test_train, y_pred_prob)
 
    from sklearn.metrics import roc_curve, roc_auc_score
    # Compute the area under the ROC curve (AUC)
    auc = roc_auc_score(Y_test_train, y_pred_prob)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='Random Forest (AUC = {:.2f})'.format(auc))
    plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # Display the ROC curve chart in Streamlit
    st.pyplot(plt)


    # Get predicted labels
    y_pred = RF.predict(X_test_train)
    # Calculate confusion matrix
    cm = confusion_matrix(Y_test_train, y_pred)
    # Plot the confusion matrix
    plt.figure()
    class_names = ['normal', 'attack']  # Specify class labels
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    ax.set_xticklabels([''] + class_names)
    ax.set_yticklabels([''] + class_names)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    # Display the confusion matrix chart in Streamlit
    st.pyplot(fig)




    st.subheader("KNN Model:")
    #### KNN Model
    KNN= KNeighborsClassifier(n_neighbors=6) 
    KNN.fit(X_train_train, Y_train_train)
    st.write("Score for the KNN Model:")
    KNN.score(X_train_train, Y_train_train), KNN.score(X_test_train, Y_test_train)

    Evaluate('KNN', KNN, X_test_train, Y_test_train)

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    # Assuming you have the trained KNN model (KNN) and the test data (X_test_train, Y_test_train)
    y_pred_prob = KNN.predict_proba(X_test_train)[:, 1]  # Predicted probabilities for the positive class
    fpr, tpr, thresholds = roc_curve(Y_test_train, y_pred_prob)
    auc = roc_auc_score(Y_test_train, y_pred_prob)

    # Plotting the ROC curve
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')  # Plotting the diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve for KNN')
    plt.legend(loc="lower right")
    plt.show()
    st.pyplot(plt)


    # Get predicted labels
    y_pred = KNN.predict(X_test_train)
    # Calculate confusion matrix
    cm = confusion_matrix(Y_test_train, y_pred)
    # Plot the confusion matrix
    plt.figure()
    class_names = ['normal', 'attack']  # Specify class labels
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    ax.set_xticklabels([''] + class_names)
    ax.set_yticklabels([''] + class_names)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    # Display the confusion matrix chart in Streamlit
    st.pyplot(fig)
   




    st.subheader("SVM Classifier:")
    #### SVM Classifier
    Linear_SVC = svm.LinearSVC(C=1)
    Linear_SVC.fit(X_train_train, Y_train_train)
    st.write("Score for the SVM Classifier:")
    Linear_SVC.score(X_train_train, Y_train_train), Linear_SVC.score(X_test_train, Y_test_train)
    Evaluate('SVM Linear SVC Kernel', Linear_SVC, X_test_train, Y_test_train)

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    # Assuming you have the trained SVM Linear SVC Kernel model (Linear_SVC) and the test data (X_test_train, Y_test_train)
    y_pred_prob = Linear_SVC.decision_function(X_test_train)  # Predicted decision scores
    fpr, tpr, thresholds = roc_curve(Y_test_train, y_pred_prob)
    auc = roc_auc_score(Y_test_train, y_pred_prob)

    # Plotting the ROC curve
 # Plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve for SVM Linear SVC Kernel')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)


    # Get predicted labels
    y_pred = Linear_SVC.predict(X_test_train)
    # Calculate confusion matrix
    cm = confusion_matrix(Y_test_train, y_pred)
    # Plot the confusion matrix
    plt.figure()
    class_names = ['normal', 'attack']  # Specify class labels
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    ax.set_xticklabels([''] + class_names)
    ax.set_yticklabels([''] + class_names)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    # Display the confusion matrix chart in Streamlit
    st.pyplot(fig)


    st.title("The Intrusion Predictor")
    Pred_Data = load_csv_file()

    # Use Pred_Data in your code
    if Pred_Data is not None:

    ## Data Encoding
       Pred_Data = pd.get_dummies(Pred_Data,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")

    # Reindex Pred_Data to match the column order of Trained_Data
       Pred_Data = Pred_Data.reindex(columns=Trained_Data.columns.drop('class'), fill_value=0)

    # Predicting the class (anomaly or attack) using the trained model
       predictions = Linear_SVC.predict(Pred_Data)

    # Convert the predicted values back to their original labels
       predicted_classes = attack_LE.inverse_transform(predictions)

    # Print the predicted classes
       st.write('The Predicted Classes are as follows:')
       st.write(predicted_classes)


    # Attach predicted_classes as a column to Pred_Data
    st.write('The Predicted class column attached with the main data frame to be predicted:')
    Pred_Data['predicted_class'] = predicted_classes


    Pred_Data




if __name__ == "__main__":
    main()