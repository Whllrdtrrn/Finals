from email import message
from multiprocessing import context
from urllib import response
from venv import create
from django.shortcuts import render,redirect
from django.template import loader
from django.contrib import auth
from django.contrib.auth import logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import HttpResponse,HttpResponseRedirect
from .models import user
from django.urls import reverse
import os
from .models import sideeffect
from .models import questioner
from django.contrib.auth.decorators import login_required
from django.urls import reverse_lazy
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.conf.urls.static import static
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import base64
from io import BytesIO
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report





# Create your views here.
def preprocessing(request):
    # request.session.clear()
    if bool(request.FILES.get('document', False)) == True:
        uploaded_file = request.FILES['document']
        name = uploaded_file.name
        request.session['name'] = name
        df = pd.read_csv(uploaded_file)
        dataFrame = df.to_json()
        request.session['df'] = dataFrame
        
        rows = len(df.index)
        request.session['rows'] = rows
        header = df.axes[1].values.tolist()
        request.session['header'] = header
        
        attributes = len(header)
        types = []
        maxs = []
        mins = []
        means = []
        # statistic attribut
        for i in range(len(header)):
            types.append(df[header[i]].dtypes)
            if df[header[i]].dtypes != 'object':
                maxs.append(df[header[i]].max())
                mins.append(df[header[i]].min())
                means.append(round(df[header[i]].mean(),2))
            else:
                maxs.append(0)
                mins.append(0)
                means.append(0)

        zipped_data = zip(header, types, maxs, mins, means)
        print(maxs)
        datas = df.values.tolist()
        data ={  
                "header": header,
                "headers": json.dumps(header),
                "name": name,
                "attributes": attributes,
                "rows": rows,
                "zipped_data": zipped_data,
                'df': datas,
                "type": types,
                "maxs": maxs,
                "mins": mins,
                "means": means,
            }
    else:
        name = 'None'
        attributes = 'None'
        rows = 'None'
        data ={
                "name": name,
                "attributes": attributes,
                "rows": rows,
            }
    return render(request, 'system/index.html', data) 

def checker_page(request):
    if request.POST:
        drop_header = request.POST.getlist('drop_header')
        print(drop_header)
        for head in drop_header:
            print(head)
        request.session['drop'] = drop_header
        method = request.POST.get('selected_method')
        if method == '1':
            return redirect('classification')
        elif method == '2':
            return redirect('clustering')
        else: 
            return redirect('preprocessing')
    else:
        return render(request, 'system/index.html')

def chooseMethod(request):
    if request.method == 'POST':
        method = request.POST.get('method')
        print('method di session : ', method)
        request.session['method'] = method
    return redirect('classification')

# def classification(request):
#     rows = request.session['rows']
#     name = request.session['name']
#     headers = request.session['header']
#     print('header : ', headers)
#     df = request.session['df']
#     df = pd.read_json(df)
#     print(df)
#     if request.session:
#         features = request.session['drop']
#         print('features : ', features)
#         method, k, graph, reportNB, reportKNN, options, crossValue, splitValue, crossValues, splitValues, outputs =  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

#         if request.session:
#             method = request.session['method']
#             print('method di class : ', method)
#             if method == '1':
#                 nameMethod = 'K-Nearest Neighbors'
#                 k = request.POST.get('knn')
#                 print(k)
#                 if request.POST.get('validation'):
#                     options = request.POST['validation']
#                     print('options selected : ', options)
#                     if options == '1':
#                         splitValue = request.POST['splitValue']
#                         print('split value : ', splitValue)
#                         reportKNN, graph = knn(df, features, options, splitValue, 0, k)
#                     elif options == '2':
#                         crossValue = request.POST['crossValue']
#                         print('cross value : ', crossValue)
#                         reportKNN, graph = knn(df, features, options, 0, crossValue,  k)
#             elif method == '2':
#                 nameMethod = 'Naive Bayes'
#                 outputs = request.POST.get('output')
#                 if request.POST.get('validation'):
#                     options = request.POST['validation']
#                     print('options selected : ', options)
#                     if options == '1':
#                         splitValue = request.POST['splitValue']
#                         print('split value : ', splitValue)
#                         reportNB, graph = naiveBayes(df, features, options, splitValue, 0, outputs)
#                     elif options == '2':
#                         crossValue = request.POST['crossValue']
#                         print('cross value : ', crossValue)
#                         reportNB, graph = naiveBayes(df, features, options, 0, crossValue, outputs)
#             if crossValue:
#                 request.session['cross'] = crossValue
#                 crossValues = request.session['cross']
#             elif splitValue:
#                 request.session['split'] = splitValue
#                 splitValues = request.session['split']

#         data = {
#             "headers": headers,
#             "method": method,
#             "naiveBayes": round((reportNB*100),2),
#             "knn": round((reportKNN*100),2),
#             "k": k,
#             "name": name,
#             "rows": rows,
#             "nameMethod": nameMethod,
#             "attributes": features,
#             "mode": options,
#             "output": outputs,
#             "splitValue": splitValues,
#             "crossValue": crossValues,
#             "confusion": graph,
#         }
#     else:
#         return redirect('preprocessing')
#     return render(request, 'classification.html', data)

def clustering(request):
    rows = request.session['rows']
    name = request.session['name']
    df = request.session['df']
    df = pd.read_json(df)
    print(df)
    features = request.session['drop']
    print(features)
    nilai_x = features[0:0]
    nilai_y = features[0:1]

    if request.method == 'POST' and request.POST['nilai_k']:
        k = request.POST['nilai_k']
        nilai_k = int(k)

        x_array = np.array(df.iloc[:, 3:5])

        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x_array)

        # Menentukan dan mengkonfigurasi fungsi kmeans
        kmeans = KMeans(n_clusters = nilai_k)
        # Menentukan kluster dari data
        kmeans.fit(x_scaled)

        # Menambahkan kolom "kluster" dalam data frame
        df['cluster'] = kmeans.labels_
        cluster = df['cluster'].value_counts()
        clusters = cluster.to_dict()
        sort_cluster = []
        label = []
        for i in sorted(clusters):
            sort_cluster.append(clusters[i])
            label.append(i)
        
        fig, ax = plt.subplots()
        sct = ax.scatter(x_scaled[:,1], x_scaled[:,0], s = 200, c = df.cluster)
        legend1 = ax.legend(*sct.legend_elements(),loc="lower left", title="Clusters")
        ax.add_artist(legend1)
        centers = kmeans.cluster_centers_
        ax.scatter(centers[:,1], centers[:,0], c='red', s=200)
        plt.title("Clustering K-Means Results")
        plt.xlabel(nilai_x)
        plt.ylabel(nilai_y)
        graph = get_graph()

        if name:
            data = {
                "name": name,
                "clusters": sort_cluster,
                "rows": rows,
                "features": features,
                "label": label,
                "chart": graph,
            }
    else:
        data = {
            "name": '',
        }

    return render(request, 'system/clustering.html', data) 

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def naiveBayes(df, features, options, size, fold, outputs):
    # Variabel independen
    fitur = features
    x = df[fitur]
    # Variabel dependen
    y = df[outputs]
    # mengubah nilai fitur menjadi rentang 0 - 1
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    from sklearn.naive_bayes import GaussianNB
    # Mengaktifkan/memanggil/membuat fungsi klasifikasi Naive Bayes
    modelnb = GaussianNB()
    train = options

    split = (int(size))/100
    cross = int(fold)

    if train == '1':
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split)
        # Memasukkan data training pada fungsi klasifikasi Naive Bayes
        nbtrain = modelnb.fit(x_train, y_train)
        # Menentukan hasil prediksi dari x_test
        y_pred = nbtrain.predict(x_test)
        ytest = np.array(y_test)
        y_test = ytest.flatten()
        report = metrics.accuracy_score(y_test, y_pred) #score prediksi

        f, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        graph = get_graph()

        classification_report(y_test, y_pred)
        print(report)
        return report, graph

    elif train == '2':
        # k - fold cross validation
        from sklearn.model_selection import cross_val_predict
        y_pred = cross_val_predict(modelnb, x, y, cv=cross)
        report = metrics.accuracy_score(y, y_pred)  #score prediksi
        
        f, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt=".0f", ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        graph = get_graph()

        classification_report(y, y_pred)
        print(report)
        return report, graph

def knn(df, features, options, size, fold, kValue):
    fitur = features
    x = df[fitur]

    y = df.iloc[:,-1:]

    # mengubah nilai fitur menjadi rentang 0 - 1
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    k = int(kValue)
    split = (int(size))/100
    cross = int(fold)

    # pemanggilan library KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k)
    train = options

    if train == '1':
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split)

        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        report = metrics.accuracy_score(y_test, y_pred) #score prediksi

        f, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        graph = get_graph()

        classification_report(y_test, y_pred)
        print(report)
        return report, graph

    elif train == '2':
        # k - fold cross validation
        from sklearn.model_selection import cross_val_predict
        y_pred = cross_val_predict(knn, x, y, cv=cross)
        report = metrics.accuracy_score(y, y_pred)  #score prediksi
        
        f, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt=".0f", ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        graph = get_graph()

        classification_report(y, y_pred)
        print(report)
        return report, graph



def services(request):
    return render (request,'services.html')

def aboutUs(request):
    return render (request,'about.html')

def contacts(request):
    return render (request,'contact.html')  
def homepage(request):
    return render (request,'homepage.html')     

def logout_view(request):
    logout(request)
    return redirect('/')


def viewDetails(request):
     userId = user.objects.get(pk=request.user.id)
     sfId = sideeffect.objects.get(pk=request.user.id)
     queId = questioner.objects.get(pk=request.user.id)
     context = {
        'userId': userId,
        'sfId': sfId,
        'queId': queId
    }
     return render (request,'view_details.html',context)

def login_page(request):
    # prod = get_object_or_404(user,id)
     if request.method == 'POST':
         user = auth.authenticate(username=request.POST['username']  ,password = request.POST['password'])
         if user is not None and user.is_superuser:
            auth.login(request,user)
            return redirect('dashboard')  
         elif user is not None and user.is_active:
                auth.login(request,user)
                # id = User.objects.all().values_list('id', flat=True).filter(username=user)
                return redirect('/server-form-page/')
                # return HttpResponse('/information-page/')          
                # return redirect('/information-page/')
         else:
            return render (request,'homepage.html', {'error':'Invalid username or password'}) 
     else:
         return render(request,'homepage.html')
    

def register_page(request):
    if request.method == 'POST':
        username = request.POST.get('username',None)
        email = request.POST.get('email',None)
        first_name = request.POST.get('first_name',None)
        last_name = request.POST.get('last_name',None)
        password = request.POST.get('password',None)
        confirm_password = request.POST.get('password_confirmation',None)

        if password==confirm_password:
            if User.objects.filter(username=username).exists():
                messages.info(request, 'Username is already taken')
                return redirect ('/') 
            elif User.objects.filter(email=email).exists():
                messages.info(request, 'Email is already taken')
                return redirect ('/') 
            else:
                user = User.objects.create_user(username=username, password=password, 
                email=email,first_name=first_name,last_name=last_name)
                user.save()
                
                return redirect('/information-page/')

        else:
            # messages.info(request, 'Both passwords are not matching')
            return render (request,'homepage.html', {'error':'Both passwords are not matching'}) 
            
    else:
        return render(request, 'homepage.html')  

def information_page (request):
    if request.method == 'POST':
                prod=user()
                prod.name = request.POST.get('name')
                prod.contact_number= request.POST.get('contact_number')
                prod.vaccination_brand = request.POST.get('vaccination_brand')
                prod.vaccination_site = request.POST.get('vaccination_site')
                prod.address = request.POST.get('address')
                prod.age = request.POST.get('age')
                prod.bday = request.POST.get('bday')
                prod.gender = request.POST.get('gender')
                if len(request.FILES) !=0:
                    prod.file = request.FILES['file']       
                prod.save()
                return redirect('/')
    return render (request,'information.html')  

def profileEdit (request):
        prod = user.objects.get(pk=request.user.id)
        if request.method == 'POST':
            if len(request.FILES) !=0:
                if len(prod.file) >0:
                    os.remove(prod.file.path)
                prod.file = request.FILES['file']       
            prod.name = request.POST.get('name')
            prod.contact_number= request.POST.get('contact_number')
            prod.vaccination_brand = request.POST.get('vaccination_brand')
            prod.vaccination_site = request.POST.get('vaccination_site')
            prod.address = request.POST.get('address')
            prod.age = request.POST.get('age')
            prod.bday = request.POST.get('bday')
            prod.gender = request.POST.get('gender')
            prod.save() 
            return redirect('/server-form-page/')
        return render (request,'profileEdit.html',{'prod':prod})    
    # try:
    #     prod = user.objects.get(pk=request.user.id)
    #     if request.method == 'POST':
    #         if len(request.FILES) !=0:
    #             if len(prod.file) >0:
    #                 os.remove(prod.file.path)
    #             prod.file = request.FILES['file']       
    #         prod.name = request.POST.get('name')
    #         prod.contact_number= request.POST.get('contact_number')
    #         prod.vaccination_brand = request.POST.get('vaccination_brand')
    #         prod.vaccination_site = request.POST.get('vaccination_site')
    #         prod.address = request.POST.get('address')
    #         prod.age = request.POST.get('age')
    #         prod.bday = request.POST.get('bday')
    #         prod.gender = request.POST.get('gender')
    #         prod.save() 
    #         return redirect('/server-form-page/')
    #     return render (request,'information.html',{'prod':prod})
    # except user.DoesNotExist:
    #     if request.method == 'POST':
    #             prod=user()
    #             prod.name = request.POST.get('name')
    #             prod.contact_number= request.POST.get('contact_number')
    #             prod.vaccination_brand = request.POST.get('vaccination_brand')
    #             prod.vaccination_site = request.POST.get('vaccination_site')
    #             prod.address = request.POST.get('address')
    #             prod.age = request.POST.get('age')
    #             prod.bday = request.POST.get('bday')
    #             prod.gender = request.POST.get('gender')
    #             if len(request.FILES) !=0:
    #                 prod.file = request.FILES['file']       
    #             prod.save()
    #             return redirect('/server-form-page/')
    #     return render (request,'information.html')  
    
            
    

# def information_page(request):    
#     if request.method == 'POST':
#          prod = user()
#          prod.name = request.POST.get('name')
#          prod.contact_number= request.POST.get('contact_number')
#          prod.vaccination_brand = request.POST.get('vaccination_brand')
#          prod.vaccination_site = request.POST.get('vaccination_site')
#          prod.address = request.POST.get('address')
#          prod.age = request.POST.get('age')
#          prod.bday = request.POST.get('bday')
#          prod.gender = request.POST.get('gender')
#          if len(request.FILES) !=0:
#              prod.file = request.FILES['file']       
#          prod.save()
         
#          return redirect('server')
#     return render (request,'information.html')

    
def sideeffect_page(request):
    try:
            prod = sideeffect.objects.get(pk=request.user.id)
            if request.method == 'POST':
                prod.muscle_ache = request.POST.get('muscle_ache')
                prod.headache = request.POST.get('headache')
                prod.fever= request.POST.get('fever')
                prod.redness = request.POST.get('redness')
                prod.swelling = request.POST.get('swelling')
                prod.tenderness = request.POST.get('tenderness')
                prod.warmth = request.POST.get('warmth')
                prod.itch = request.POST.get('itch')
                prod.induration= request.POST.get('induration')
                prod.feverish = request.POST.get('feverish')
                prod.chills= request.POST.get('chills')
                prod.join_pain = request.POST.get('join_pain')
                prod.fatigue= request.POST.get('fatigue')
                prod.nausea= request.POST.get('nausea')
                prod.vomiting = request.POST.get('vomiting')
                prod.save()
                messages.success(request, "Successfully Submitted")
                return redirect('/')
                # return render (request,'login_page.html')
            else:               
                return render (request,'sideeffect.html',{'prod':prod})
    except sideeffect.DoesNotExist:        
            prod = sideeffect()
            if request.method == 'POST':
                prod.muscle_ache = request.POST.get('muscle_ache')
                prod.headache = request.POST.get('headache')
                prod.fever= request.POST.get('fever')
                prod.redness = request.POST.get('redness')
                prod.swelling = request.POST.get('swelling')
                prod.tenderness = request.POST.get('tenderness')
                prod.warmth = request.POST.get('warmth')
                prod.itch = request.POST.get('itch')
                prod.induration= request.POST.get('induration')
                prod.feverish = request.POST.get('feverish')
                prod.chills= request.POST.get('chills')
                prod.join_pain = request.POST.get('join_pain')
                prod.fatigue= request.POST.get('fatigue')
                prod.nausea= request.POST.get('nausea')
                prod.vomiting = request.POST.get('vomiting')
                prod.save()
                messages.success(request, "Successfully Submitted")
                return redirect('/')
            else:             
                 return render (request,'sideeffect.html')
   
def server_form(request):
    try:
        prod = questioner.objects.get(pk=request.user.id)
        if request.method == 'POST':
            prod.Q0 = request.POST.get('Q0')
            prod.Q1 = request.POST.get('Q1')
            prod.Q2 = request.POST.get('Q2')
            prod.Q3 = request.POST.get('Q3')
            prod.Q4 = request.POST.get('Q4')
            prod.Q5 = request.POST.get('Q5')
            prod.Q6 = request.POST.get('Q6')
            prod.Q7 = request.POST.get('Q7')
            prod.Q8= request.POST.get('Q8')
            prod.Q9 = request.POST.get('Q9')
            prod.Q10= request.POST.get('Q10')
            prod.Q11 = request.POST.get('Q11')
            prod.Q12= request.POST.get('Q12')
            prod.Q13= request.POST.get('Q13')
            prod.Q14 = request.POST.get('Q14')
            prod.Q15= request.POST.get('Q15')
            prod.Q16 = request.POST.get('Q16')
            prod.Q17= request.POST.get('Q17')
            prod.Q18= request.POST.get('Q18')
            prod.Q19 = request.POST.get('Q19')
            prod.Q20 = request.POST.get('Q20')
            prod.Q21 = request.POST.get('Q21')
            prod.Q22 = request.POST.get('Q22')
            prod.allergy = request.POST.get('allergy')
            prod.allergy1 = request.POST.get('allergy1')
            prod.allergy2 = request.POST.get('allergy2')
            prod.allergy3 = request.POST.get('allergy3')
            prod.allergy4 = request.POST.get('allergy4')
            prod.allergy5 = request.POST.get('allergy5')            
            prod.Q23 = request.POST.get('Q23')
            prod.Q24 = request.POST.get('Q24')
            #    item = questioner(Q0=Q0,Q1=Q1,Q2=Q2,Q3=Q3,Q4=Q4,Q5=Q5,Q6=Q6,Q7=Q7,Q8=Q8,Q9=Q9,Q10=Q10,Q11=Q11,
            #     Q12=Q12,Q13=Q13,Q14=Q14,Q15=Q15,Q16=Q16,Q17=Q17,Q18=Q18,Q19=Q19,Q20=Q20,Q21=Q21,Q22=Q22,allergy=allergy,
            #     Q23=Q23,Q24=Q24)
            prod.save()
            return redirect('/side-effect-page/')
        else:   
            return render (request,'serverform.html',{'prod':prod}) 

    except questioner.DoesNotExist:
        prod = questioner()
        if request.method == 'POST':
            prod.Q0 = request.POST.get('Q0')
            prod.Q1 = request.POST.get('Q1')
            prod.Q2 = request.POST.get('Q2')
            prod.Q3 = request.POST.get('Q3')
            prod.Q4 = request.POST.get('Q4')
            prod.Q5 = request.POST.get('Q5')
            prod.Q6 = request.POST.get('Q6')
            prod.Q7 = request.POST.get('Q7')
            prod.Q8= request.POST.get('Q8')
            prod.Q9 = request.POST.get('Q9')
            prod.Q10= request.POST.get('Q10')
            prod.Q11 = request.POST.get('Q11')
            prod.Q12= request.POST.get('Q12')
            prod.Q13= request.POST.get('Q13')
            prod.Q14 = request.POST.get('Q14')
            prod.Q15= request.POST.get('Q15')
            prod.Q16 = request.POST.get('Q16')
            prod.Q17= request.POST.get('Q17')
            prod.Q18= request.POST.get('Q18')
            prod.Q19 = request.POST.get('Q19')
            prod.Q20 = request.POST.get('Q20')
            prod.Q21 = request.POST.get('Q21')
            prod.Q22 = request.POST.get('Q22')
            prod.allergy = request.POST.get('allergy')
            prod.allergy1 = request.POST.get('allergy1')
            prod.allergy2 = request.POST.get('allergy2')
            prod.allergy3 = request.POST.get('allergy3')
            prod.allergy4 = request.POST.get('allergy4')
            prod.allergy5 = request.POST.get('allergy5')
            prod.Q23 = request.POST.get('Q23')
            prod.Q24 = request.POST.get('Q24')
            #    item = questioner(Q0=Q0,Q1=Q1,Q2=Q2,Q3=Q3,Q4=Q4,Q5=Q5,Q6=Q6,Q7=Q7,Q8=Q8,Q9=Q9,Q10=Q10,Q11=Q11,
            #     Q12=Q12,Q13=Q13,Q14=Q14,Q15=Q15,Q16=Q16,Q17=Q17,Q18=Q18,Q19=Q19,Q20=Q20,Q21=Q21,Q22=Q22,allergy=allergy,
            #     Q23=Q23,Q24=Q24)
            prod.save()
            return redirect('/side-effect-page/')
        else:   
            return render (request,'serverform.html')                  

def success_page (request):
     
     template = loader.get_template('success.html')
     return HttpResponse(template.render()) 

def toggle_status(request,id):
    status = User.objects.get(id=id)
    status.is_active = 0
    status.save()
    return redirect ('/dashboard/')

def toggle_status_active(request,id):
    status = User.objects.get(id=id)
    status.is_active = 1
    status.save()
    return redirect ('/dashboard/')    

def dashboard (request):
     item_list = user.objects.all().values()
     item_lists = sideeffect.objects.all().values()
     quesT = questioner.objects.all().values()
     userAccount = User.objects.all().values().filter(is_superuser=False)
     total_user = userAccount.count()
     total_admin = User.objects.filter(is_superuser=True).count()
    
     context = {
        'item_list': item_list,
        'item_lists': item_lists,
        'quesT': quesT,
        'userAccount': userAccount,
        'total_user':total_user,
        'total_admin':total_admin,
    }
   
     return render (request,'system/dashboard.html',context)

    # template = loader.get_template('system/dashboard.html')
    # return HttpResponse(template.render()) 
    
# def delete(request):
#     prod = User.objects.get(pk=request.user.id)
#     prod.is_active = True
#     prod.save()
#     return render(request,'system/dashboard.html')     
# def delete(request):
#    prod = questioner.objects.get(pk=request.user.id)
#    prod.delete()
#    return redirect (request, 'dashboard/')
# def edit(request,id):
#     users = user.objects.get(pk=id)
#     form = userForm(request.POST or None,instance = users)
#     return render(request,'information.html',{'form':form })   