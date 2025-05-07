from tkinter import *
import tkinter
import pandas as pd
import networkx as nx
from tqdm import tqdm
import collections
from multiprocessing import Pool
import time
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tkinter import filedialog
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

main = tkinter.Tk()
main.title("BotChase") #designing main screen
main.geometry("1300x1200")

global filename
global ctuDataset
global dg, dict_nodes_list, ip_nodes_list
global between_ness, clustering, alpha_cent
global cm


def upload():
    global filename
    global ctuDataset
    filename = filedialog.askopenfilename(initialdir="CTU-13-Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    ctuDataset = pd.read_csv(filename)
    text.insert(END,'Dataset size  : \n\n')
    text.insert(END,'Total Rows    : '+str(ctuDataset.shape[0])+"\n")
    text.insert(END,'Total Columns : '+str(ctuDataset.shape[1])+"\n\n")
    text.insert(END,'Dataset Samples\n\n')
    text.insert(END,str(ctuDataset.head())+"\n\n")
    
def kmeans():
    global ctuDataset
    text.delete('1.0', END)
    
    # Filter Botnet and Benign records
    ctu_bot = ctuDataset[ctuDataset['Label'].str.contains('Botnet')]
    ctu_benign = ctuDataset[~ctuDataset['Label'].str.contains('Botnet')]

    # Balance the dataset
    ctu_benign = ctu_benign.sample(n=min(ctu_bot.shape[0], ctu_benign.shape[0]), random_state=0)
    
    # Merge botnet and sampled benign records
    ctuDataset = pd.concat([ctu_bot, ctu_benign])

    text.insert(END, 'Dataset size before filtering:\n\n')
    text.insert(END, f'Total Rows    : {ctuDataset.shape[0]}\n')
    text.insert(END, f'Total Columns : {ctuDataset.shape[1]}\n\n')

    # Select features for clustering
    X = ctuDataset[['Dur', 'TotPkts', 'TotBytes', 'SrcBytes']]

    # Apply KMeans
    kmeans = KMeans(n_clusters=2, random_state=0)
    ctuDataset['Cluster'] = kmeans.fit_predict(X)

    text.insert(END, 'Dataset size after filtering:\n\n')
    text.insert(END, f'Total Rows    : {ctuDataset.shape[0]}\n')
    text.insert(END, f'Total Columns : {ctuDataset.shape[1]}\n\n')
    text.insert(END, 'KMeans clustering applied successfully.\n')
    

#get nodes chunk
def nodeOrder(nodes, order):
    node_order = iter(nodes)
    while nodes:
        name = tuple(itertools.islice(node_order, order))
        if not name:
            return
        yield name

#return betweenness
def betweenmap(Graph_normalized_weight_sources_tuple):
    return nx.betweenness_centrality_source(*Graph_normalized_weight_sources_tuple)


def betweennessCentrality(Graphs, processes=None):
    p = Pool(processes=processes)
    nodes_divisor = len(p._pool) * 2
    nodes_chunks = list(nodeOrder(Graphs.nodes(), int(Graphs.order() / nodes_divisor)))
    numb_chunks = len(nodes_chunks)
    between_sc = p.map(betweenmap,
                  zip([Graphs] * numb_chunks,
                      [True] * numb_chunks,
                      [True] * numb_chunks,
                      nodes_chunks))

    # Reduce the partial solutions
    between_c = between_sc[0]
    for between in between_sc[1:]:
        for n in between:
            between_c[n] += between[n]
    return between_c


def graphTransform():
    global ctuDataset
    global dg, dict_nodes_list, ip_nodes_list
    global between_ness, clustering, alpha_cent
    text.delete('1.0', END)

    duplicates_RowsDF = ctuDataset[ctuDataset.duplicated(['SrcAddr', 'DstAddr'])]
    duplicates_Rows = list(duplicates_RowsDF.index.values)
    ctuDataset = ctuDataset.drop(duplicates_Rows)

    ctu_temp = pd.merge(ctuDataset, duplicates_RowsDF, how='inner', on=['SrcAddr', 'DstAddr'])

    sum_columns = ctu_temp['TotPkts_x'] + ctu_temp['TotPkts_y']
    ctu_temp['TotPkts'] = sum_columns
    ctuDataset = ctu_temp[['SrcAddr', 'DstAddr', 'TotPkts']]

    #graph building starts here
    dg = nx.DiGraph()

    #deleting duplicate ip address to have only one vertex for each IP
    source_address_list = list(ctuDataset['SrcAddr'])
    source_address_list.extend(list(ctuDataset['DstAddr']))
    ip_nodes_list = list(set(source_address_list))
    dg.add_nodes_from(ip_nodes_list)

    dict_nodes_list = collections.defaultdict(dict)

    for index, row in tqdm(ctuDataset.iterrows(), total=ctuDataset.shape[0], desc="{Generating bot graph}"):
        e = (row['SrcAddr'], row['DstAddr'], row['TotPkts'])
    
        if dg.has_edge(*e[:2]):
            edgeData = dg.get_edge_data(*e)
            weight = edgeData['weight']
            dg.add_weighted_edges_from([(row['SrcAddr'], row['DstAddr'], row['TotPkts'] + weight)])
            dict_nodes_list[row['SrcAddr']]['out-degree-weight'] += row['TotPkts']
            dict_nodes_list[row['DstAddr']]['in-degree-weight'] += row['TotPkts']
            dict_nodes_list[row['SrcAddr']]['out-degree'] += 1
            dict_nodes_list[row['DstAddr']]['in-degree'] += 1
        else:
            dg.add_weighted_edges_from([(row['SrcAddr'], row['DstAddr'], row['TotPkts'])])
            dict_nodes_list[row['SrcAddr']]['out-degree-weight'] = row['TotPkts']
            dict_nodes_list[row['SrcAddr']]['in-degree-weight'] = 0
            dict_nodes_list[row['DstAddr']]['in-degree-weight'] = row['TotPkts']
            dict_nodes_list[row['DstAddr']]['out-degree-weight'] = 0
            dict_nodes_list[row['SrcAddr']]['out-degree'] = 1
            dict_nodes_list[row['SrcAddr']]['in-degree'] = 0
            dict_nodes_list[row['DstAddr']]['in-degree'] = 1
            dict_nodes_list[row['DstAddr']]['out-degree'] = 0

    text.insert(END,'Number of nodes: ' + str(nx.number_of_nodes(dg))+"\n\n")
    text.insert(END,'Number of edges: ' + str(nx.number_of_edges(dg))+"\n\n")
    text.insert(END,'Network graph created\n\n')

    text.insert(END,'Betweeness centrality time calculation for all ip address or nodes\n\n')
    start_time = time.time()
    between_ness = betweennessCentrality(dg, 2)
    text.insert(END,"Execution Time : "+str((time.time() - start_time))+"\n\n")

    #clustering time
    text.insert(END,'Clustering time calculation\n\n')
    start_time = time.time()
    clustering = nx.clustering(dg, weight='weight')
    text.insert(END,"Clustering Time : "+str((time.time() - start_time))+"\n\n")

    #Alpha Centrality time calculation
    text.insert(END,'Alpha Centrality time calculation')
    start_time = time.time()
    alpha_cent = nx.algorithms.centrality.katz_centrality_numpy(dg, weight='weight')
    text.insert(END,"Alpha Centrality Time : "+str((time.time() - start_time))+"\n\n")

    #bot_list = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204', '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']

    for i in dict_nodes_list:
        #print(dict_nodes_list[i]['out-degree'])
        if dict_nodes_list[i]['out-degree'] > 10:
            print(dict_nodes_list[i]['out-degree'])
            dict_nodes_list[i]['bot'] = 1
        else:
            dict_nodes_list[i]['bot'] = 0

        dict_nodes_list[i]['bc'] = between_ness[i]
        dict_nodes_list[i]['lcc'] = clustering[i]
        dict_nodes_list[i]['ac'] = alpha_cent[i]
    
def featuresNormalization():
    text.delete('1.0', END)
    for nodes in tqdm(ip_nodes_list, desc="{features Normalization module}"):
        counter = 0 #N is a counter for the total neighbors of each node with D=2
        in_degree = 0 #s is the total sum of all the features of the node's neighbors
        out_degree = 0
        in_degree_weight = 0
        out_degree_weight = 0
        s_between = 0
        s_clustering = 0
        s_alpha = 0
    
        for neighbors in dg.neighbors(nodes):
            in_degree += dict_nodes_list[neighbors]['in-degree'] 
            out_degree += dict_nodes_list[neighbors]['out-degree']
            in_degree_weight += dict_nodes_list[neighbors]['in-degree-weight']
            out_degree_weight += dict_nodes_list[neighbors]['out-degree-weight']
            s_between += between_ness[neighbors]
            s_clustering += clustering[neighbors]
            s_alpha += alpha_cent[neighbors]
            counter += 1
        
            for n in dg.neighbors(neighbors):
                in_degree += dict_nodes_list[neighbors]['in-degree'] 
                out_degree += dict_nodes_list[neighbors]['out-degree']
                in_degree_weight += dict_nodes_list[neighbors]['in-degree-weight']
                out_degree_weight += dict_nodes_list[neighbors]['out-degree-weight']
                s_between += between_ness[neighbors]
                s_clustering += clustering[neighbors]
                s_alpha += alpha_cent[neighbors]
                counter += 1
   
        if counter != 0:
            if in_degree != 0:
                dict_nodes_list[nodes]['in-degree'] = dict_nodes_list[nodes]['in-degree'] / (in_degree/counter)
        
            if out_degree != 0:
                dict_nodes_list[nodes]['out-degree'] = dict_nodes_list[nodes]['out-degree'] / (out_degree/counter)
        
            if in_degree_weight != 0:
                dict_nodes_list[nodes]['in-degree-weight'] = dict_nodes_list[nodes]['in-degree-weight'] / (in_degree_weight/counter)
        
            if out_degree_weight != 0:
                dict_nodes_list[nodes]['out-degree-weight'] = dict_nodes_list[nodes]['out-degree-weight'] / (out_degree_weight/counter)
        
            if s_between != 0:
                dict_nodes_list[nodes]['bc'] = dict_nodes_list[nodes]['bc'] / (s_between/counter)
        
            if s_clustering != 0:
                dict_nodes_list[nodes]['lcc'] = dict_nodes_list[nodes]['lcc'] / (s_clustering/counter)
        
            if s_alpha != 0:
                dict_nodes_list[nodes]['ac'] = dict_nodes_list[nodes]['ac'] / (s_alpha/counter)
        
        
    text.insert(END,'Normalizing features process completed & below are some sample records\n\n')        
    graph_df = pd.DataFrame.from_dict(dict_nodes_list, orient='index')
    text.insert(END,str(graph_df.head())+"\n\n")
    graph_df.to_csv('normalize_data.csv', index=True)
    text.insert(END,'Normalized & transformed data saved inside normalize_data.csv file\n\n')

def Performance_evaluation(algorithum,y_test,predict):

    unique_labels=['Normal Pattern', 'Bot Pattern']
    p = precision_score(y_test, predict, average='macro', zero_division=0) * 100
    r = recall_score(y_test, predict, average='macro', zero_division=0) * 100
    f = f1_score(y_test, predict, average='macro', zero_division=0) * 100
    a = accuracy_score(y_test, predict) * 100

    # Display precision, recall, F1-score, and accuracy in the Text widget
    text.insert(END, f"{algorithum} Precision: " + str(p) + "\n")
    text.insert(END, f"{algorithum} Recall: " + str(r) + "\n")
    text.insert(END, f"{algorithum} FMeasure: " + str(f) + "\n")
    text.insert(END, f"{algorithum} Accuracy: " + str(a) + "\n\n")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, predict)
    
    # Compute classification report
    report = classification_report(y_test, predict, target_names=unique_labels)
    text.insert(END, f"{algorithum} Classification Report:\n")
    text.insert(END, report)    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title(f'{algorithum} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
def Train_Test_Splitting():
    text.delete('1.0', END)
    global  X_train, X_test, Y_train, Y_test
    df = pd.read_csv('normalize_data.csv')
    text.insert(END,'Normalized data loading to decision tree classifier\n\n')
    X = df[['out-degree-weight', 'in-degree-weight', 'out-degree', 'in-degree', 'bc', 'lcc', 'ac']]
    Y = df['bot']
    Y = Y.tolist()
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END,'Total dataset size to build model : '+str(X.shape)+"\n\n")
    text.insert(END,'Model training records size       : '+str(X_train.shape)+"\n\n")
    text.insert(END,'Model testing records size        : '+str(X_test.shape)+"\n\n")



def naiveBayes():
    text.delete('1.0', END)
    global X_train, X_test, Y_train, Y_test

    from sklearn.naive_bayes import GaussianNB
    cls = GaussianNB()
    cls = cls.fit(X_train, Y_train)
    y_pred = cls.predict(X_test)

    Performance_evaluation("Existing Naive Bayes", Y_test, y_pred)

# LDA Classifier
def ldaClassifier():
    text.delete('1.0', END)
    global X_train, X_test, Y_train, Y_test

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    cls = LinearDiscriminantAnalysis()
    cls = cls.fit(X_train, Y_train)
    y_pred = cls.predict(X_test)

    Performance_evaluation("Existing LDA", Y_test, y_pred)

# Ridge Classifier
def ridgeClassifier():
    text.delete('1.0', END)
    global X_train, X_test, Y_train, Y_test

    from sklearn.linear_model import RidgeClassifier
    cls = RidgeClassifier()
    cls = cls.fit(X_train, Y_train)
    y_pred = cls.predict(X_test)

    Performance_evaluation("Existing Ridge Classifier", Y_test, y_pred)

# KNN Classifier
def knnClassifier():
    text.delete('1.0', END)
    global X_train, X_test, Y_train, Y_test

    from sklearn.neighbors import KNeighborsClassifier
    cls = KNeighborsClassifier()
    cls = cls.fit(X_train, Y_train)
    y_pred = cls.predict(X_test)

    Performance_evaluation("Existing KNN", Y_test, y_pred)

# SVM Classifier
def svmClassifier():
    text.delete('1.0', END)
    global X_train, X_test, Y_train, Y_test

    from sklearn.svm import SVC
    cls = SVC()
    cls = cls.fit(X_train, Y_train)
    y_pred = cls.predict(X_test)

    Performance_evaluation("Existing SVM", Y_test, y_pred)
    
def decisionTree():
    text.delete('1.0', END)
    global  clf,X_train, X_test, Y_train, Y_test

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,Y_train)
    y_pred = clf.predict(X_test)

    Performance_evaluation("Proposed DTC",Y_test,y_pred)

def Prediction():
    global clf
    unique_labels=['Normal Pattern', 'Bot Pattern']

    filename = filedialog.askopenfilename(initialdir=".")
    text.delete('1.0', END)
    text.insert(END, f'{filename} Loaded\n')
    test = pd.read_csv(filename)
    test1= test[['out-degree-weight', 'in-degree-weight', 'out-degree', 'in-degree', 'bc', 'lcc', 'ac']]

    # Assuming 'clf' is your classifier model
    predict = clf.predict(test1)
    
    # Iterate through each row of the dataset and print its corresponding predicted outcome
    text.insert(END, f'Predicted Outcomes for each row:\n')
    for index, row in test1.iterrows():
        # Get the prediction for the current row
        predicted_index = predict[index]
        
        # Map predicted index to its corresponding label using unique_labels_list
        predicted_outcome = unique_labels[predicted_index]
        
        # Print the current row of the dataset followed by its predicted outcome
        text.insert(END, f'Row {index + 1}: {row.to_dict()} - Predicted Outcome: {predicted_outcome}\n\n\n\n\n')
        
        
def close():
  global main
  main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='ML Enabled Intelligent Bot Detection in Network Communications')
title.config(bg='dark sea green', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=130)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=100)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload CTU Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

kmeansButton = Button(main, text="Apply Unsupervised Learning (K-means)", command=kmeans)
kmeansButton.place(x=50,y=150)
kmeansButton.config(font=font1) 

transformButton = Button(main, text="Flow Ingestion & Graph Transformation", command=graphTransform)
transformButton.place(x=50,y=200)
transformButton.config(font=font1) 

normalizationButton = Button(main, text="Features Extraction & Normalization", command=featuresNormalization)
normalizationButton.place(x=50,y=250)
normalizationButton.config(font=font1) 


dtButton = Button(main, text="Train Test Splitting ", command=Train_Test_Splitting)
dtButton.place(x=50,y=300)
dtButton.config(font=font1)

dtButton = Button(main, text="Existing NBC ", command=naiveBayes)
dtButton.place(x=50,y=350)
dtButton.config(font=font1)

dtButton = Button(main, text="Existing LDA ", command=ldaClassifier)
dtButton.place(x=50,y=400)
dtButton.config(font=font1)

dtButton = Button(main, text="Existing Ridge ", command=ridgeClassifier)
dtButton.place(x=50,y=450)
dtButton.config(font=font1)

dtButton = Button(main, text="Existing KNN ", command=knnClassifier)
dtButton.place(x=50,y=500)
dtButton.config(font=font1)

dtButton = Button(main, text="Existing SVM ", command=svmClassifier)
dtButton.place(x=50,y=550)
dtButton.config(font=font1)

dtButton = Button(main, text="Proposed DTC", command=decisionTree)
dtButton.place(x=50,y=600)
dtButton.config(font=font1)

dtButton = Button(main, text="Prediction From Test Data", command=Prediction)
dtButton.place(x=50,y=650)
dtButton.config(font=font1)

graphButton = Button(main, text="Exit", command=close)
graphButton.place(x=50,y=700)
graphButton.config(font=font1)



main.config(bg='dark sea green')
main.mainloop()
