from torch.utils.data import DataLoader
import torch
from email import message
from multiprocessing import context
from urllib import response
from venv import create

from django.shortcuts import render, redirect
from django.template import loader
from django.contrib import auth
from django.contrib.auth import logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import HttpResponse, HttpResponseRedirect
from .models import user
from django.urls import reverse
import os

from .filters import OrderFilter
from .models import sideeffect
from .models import questioner
from django.contrib.auth.decorators import login_required
from django.urls import reverse_lazy
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.conf.urls.static import static
# import tensorflow as tf
import pandas as pd
from pandas.api.types import is_object_dtype
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import base64
from io import BytesIO
from django.utils.timezone import now
from django.db.models import Q

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# email verification
from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.template.loader import render_to_string
from .tokens import account_activation_token
from django.core.mail import EmailMessage
from django.shortcuts import get_object_or_404
import warnings
warnings.filterwarnings('ignore')


# from google.colab import drive # for google drive access
# drive.mount('/content/drive')


# Create your views here.
# def preprocessingOOOO(request):

#     if bool(request.FILES.get('document', False)) == True:
#         uploaded_file = request.FILES['document']
#         name = uploaded_file.name
#         request.session['name'] = name
#         df = pd.read_csv(uploaded_file)
#         dataFrame = df.to_json()
#         request.session['df'] = dataFrame

#         rows = len(df.index)
#         request.session['rows'] = rows
#         header = df.axes[1].values.tolist()
#         request.session['header'] = header

#         attributes = len(header)
#         types = []
#         maxs = []
#         mins = []
#         means = []
#         # statistic attribut
#         for i in range(len(header)):
#             types.append(df[header[i]].dtypes)
#             if df[header[i]].dtypes != 'object':
#                 maxs.append(df[header[i]].max())
#                 mins.append(df[header[i]].min())
#                 means.append(round(df[header[i]].mean(),2))
#             else:
#                 maxs.append(0)
#                 mins.append(0)
#                 means.append(0)

#         zipped_data = zip(header, types, maxs, mins, means)
#         print(maxs)
#         datas = df.values.tolist()
#         data ={
#                 "header": header,
#                 "headers": json.dumps(header),
#                 "name": name,
#                 "attributes": attributes,
#                 "rows": rows,
#                 "zipped_data": zipped_data,
#                 'df': datas,
#                 "type": types,
#                 "maxs": maxs,
#                 "mins": mins,
#                 "means": means,
#             }
#     else:
#         name = 'None'
#         attributes = 'None'
#         rows = 'None'
#         data ={
#                 "name": name,
#                 "attributes": attributes,
#                 "rows": rows,
#             }
#     return render(request, 'system/index.html', data)
def LayoutHome(request):
    return render(request, 'Layout/LayoutHome.html')


def LayoutIndex(request):
    prod = user()
    side = sideeffect()
    ques = questioner()
    if request.method == 'POST':
        prod.name = request.POST.get('name')
        prod.contact_number = request.POST.get('contact_number')
        prod.vaccination_brand = request.POST.get('vaccination_brand')
        prod.vaccination_site = request.POST.get('vaccination_site')
        prod.address = request.POST.get('address')
        prod.age = request.POST.get('age')
        prod.bday = request.POST.get('bday')
        prod.gender = request.POST.get('gender')
        side.muscle_ache = request.POST.get('muscle_ache')
        side.headache = request.POST.get('headache')
        side.fever = request.POST.get('fever')
        side.redness = request.POST.get('redness')
        side.swelling = request.POST.get('swelling')
        side.tenderness = request.POST.get('tenderness')
        side.warmth = request.POST.get('warmth')
        side.itch = request.POST.get('itch')
        side.induration = request.POST.get('induration')
        side.feverish = request.POST.get('feverish')
        side.chills = request.POST.get('chills')
        side.join_pain = request.POST.get('join_pain')
        side.fatigue = request.POST.get('fatigue')
        side.nausea = request.POST.get('nausea')
        side.vomiting = request.POST.get('vomiting')
        ques.Q0 = request.POST.get('Q0')
        ques.Q1 = request.POST.get('Q1')
        ques.Q2 = request.POST.get('Q2')
        ques.Q3 = request.POST.get('Q3')
        ques.Q4 = request.POST.get('Q4')
        ques.Q5 = request.POST.get('Q5')
        ques.Q6 = request.POST.get('Q6')
        ques.Q7 = request.POST.get('Q7')
        ques.Q8 = request.POST.get('Q8')
        ques.Q9 = request.POST.get('Q9')
        ques.Q10 = request.POST.get('Q10')
        ques.Q11 = request.POST.get('Q11')
        ques.Q12 = request.POST.get('Q12')
        ques.Q13 = request.POST.get('Q13')
        ques.Q14 = request.POST.get('Q14')
        ques.Q15 = request.POST.get('Q15')
        ques.Q16 = request.POST.get('Q16')
        ques.Q17 = request.POST.get('Q17')
        ques.Q18 = request.POST.get('Q18')
        ques.Q19 = request.POST.get('Q19')
        ques.Q20 = request.POST.get('Q20')
        ques.Q21 = request.POST.get('Q21')
        ques.Q22 = request.POST.get('Q22')
        ques.allergy = request.POST.get('allergy')
        ques.allergy1 = request.POST.get('allergy1')
        ques.allergy2 = request.POST.get('allergy2')
        ques.allergy3 = request.POST.get('allergy3')
        ques.allergy4 = request.POST.get('allergy4')
        ques.allergy5 = request.POST.get('allergy5')
        ques.Q23 = request.POST.get('Q23')
        ques.Q24 = request.POST.get('Q24')
        if len(request.FILES) != 0:
            prod.file = request.FILES['file']
        prod.save()
        side.save()
        ques.save()

        messages.success(request, "Successfully Submitted")
        return redirect('/')
    return render(request, 'Layout/LayoutIndex.html')


def preprocessing(data):
    # Palitan ng read something sql
    ds = pd.read_csv(data, encoding="unicode_escape")

    # Drop all column's NAs except NAs on symptoms
    ds = ds.dropna(subset=["SYMPTOM1", "SYMPTOM2",
                   "SYMPTOM3", "SYMPTOM4", "SYMPTOM5"])

    # Get all unique symptom
    symptoms = list(set([x for i in range(1, 6)
                    for x in ds[f"SYMPTOM{i}"].unique() if x is not np.nan]))

    # Combine all symptom column to one
    ds_copy_symptom = ds[["SYMPTOM1", "SYMPTOM2", "SYMPTOM3",
                          "SYMPTOM4", "SYMPTOM5"]].astype(str).agg(', '.join, axis=1)

    # Get important/target features
    ds = ds[["SEX", "ALLERGIES", "CUR_ILL",
             "HISTORY", "RECOVD", "VAX_NAME-Unique"]]

    # Replace values for recovered
    ds['SEX'] = ds['SEX'].apply(
        lambda x: 0 if x == "F" else 1 if x == "M" else 2 if x == "U" else None)

    # Replace values for allergies
    ds['ALLERGIES'] = ds['ALLERGIES'].apply(
        lambda x: 1 if x is not None else 0)

    # Replace values for current illness
    ds['CUR_ILL'] = ds['CUR_ILL'].apply(lambda x: 1 if x is not None else 0)

    # Replace values for recovered
    ds['HISTORY'] = ds['HISTORY'].apply(lambda x: 1 if x is not None else 0)

    # Replace values for recovered
    ds['RECOVD'] = ds['RECOVD'].apply(
        lambda x: "Yes" if x == "Y" else "No" if x == "N" else "U" if x == "NaN" else None)

    # Expand Symptom
    symptoms_df = pd.DataFrame()
    for symptom in symptoms:
        symptoms_df[symptom] = (ds_copy_symptom.str.contains(symptom))
    symptoms_df = symptoms_df.fillna(0)
    # symptoms.value_counts()
    print(symptoms_df.shape)
    print(ds.shape)

    # Combine symptoms and important features
    ds = pd.concat([ds, symptoms_df], axis=1)

    # ds = ds.dropna()

    # Discretize Categorical values
    colToLabel = []
    for col in ds.columns:
        if is_object_dtype(ds[col]):
            colToLabel.append(col)
    le = LabelEncoder()
    labels = {}
    for col in colToLabel:
        ds[col] = le.fit_transform(ds[col])
        labels[col] = dict(zip(le.classes_, range(len(le.classes_))))

    return ds


def build_model(data):
    ############# MODEL #################
    inertias = []  # wcss
    prev_y = 0
    highest = 0

    # Elbow method
    for x in range(1, 11):
        kmeans = KMeans(n_clusters=x, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        y = kmeans.inertia_
        inertias.append(y)
        print(y)

        if len(inertias) > 1:
            # Get the highest subsequent difference to be valued as k
            diff = abs(inertias[-2] - inertias[-1])
            if diff > highest:
                k = x
        # Get the highest subsequent difference to be valued as k

    # norm_distances = [float(i)/max(distances) for i in distances]
    # k = np.argmin(norm_distances) + 1

    # Set a final model with fixed k
    final_kmeans = KMeans(n_clusters=3, init='k-means++',
                          max_iter=300, n_init=10, random_state=0)

    return final_kmeans, inertias


def apply_scaler_pca(data):
    print(len(data))
    x_std = MinMaxScaler().fit_transform(data)

    pca = PCA(n_components=2)
    x_pca = pd.DataFrame(pca.fit_transform(x_std))
    print(len(x_pca))
    return x_pca


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


def clustering(request):
    df = preprocessing("PreProcToAlgoVaers2020.csv")
    x_scaled = apply_scaler_pca(df)
    kmeans, inertia = build_model(x_scaled)

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
    sct = ax.scatter(x_scaled[0], x_scaled[1], s=200, c=df.cluster)
    legend1 = ax.legend(*sct.legend_elements(),
                        loc="lower left", title="Clusters")
    ax.add_artist(legend1)
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200)
    plt.title("Clustering K-Means Results")
    plt.xlabel("Pca 1")
    plt.ylabel("Symptoms/pca 2")
    graph = get_graph()

# if name:
    data = {
        # "name": name,
        "clusters": sort_cluster,
        # "rows": rows,
        # "features": features,
        "label": label,
        "chart": graph,
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


def services(request):
    return render(request, 'services.html')


def aboutUs(request):
    return render(request, 'about.html')


def contacts(request):
    return render(request, 'contact.html')


def verification(request):
    return render(request, 'Email_confirmation.html')


def home_page(request):
    return render(request, 'homepage.html')


def logout_view(request):
    logout(request)
    messages.success(request, "Successfully Logout!!")
    return redirect('/')


def viewDetails(request):
    userId = user.objects.get(author=request.user)
    sfId = sideeffect.objects.get(author=request.user)
    queId = questioner.objects.get(author=request.user)
    context = {
        'userId': userId,
        'sfId': sfId,
        'queId': queId
    }
    return render(request, 'view_details.html', context)


def login_page(request):
    # prod = get_object_or_404(user,id)
    if request.method == 'POST':
        user = auth.authenticate(
            username=request.POST['username'], password=request.POST['password'])
        if user is not None and user.is_superuser:
            auth.login(request, user)
            return redirect('/dashboard/')
        elif user is not None and user.is_active:
            auth.login(request, user)
            # id = User.objects.all().values_list('id', flat=True).filter(username=user)
            return redirect('/information-page/')
            # return HttpResponse('/information-page/')
            # return redirect('/information-page/')
        else:
            return render(request, 'homepage.html', {'error': 'Invalid username or password'})
    else:
        return render(request, 'homepage.html')


def register_page(request):
    if request.method == 'POST':
        username = request.POST.get('username', None)
        email = request.POST.get('email', None)
        first_name = request.POST.get('first_name', None)
        last_name = request.POST.get('last_name', None)
        password = request.POST.get('password', None)
        confirm_password = request.POST.get('password_confirmation', None)

        if password == confirm_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username is already taken')
                return redirect('/')
            elif User.objects.filter(email=email).exists():
                messages.error(request, 'Email is already taken')
                return redirect('/')
            else:
                user = User.objects.create_user(username=username, password=password, email=email, first_name=first_name, last_name=last_name)
                user.save()
                return redirect('/loginpage/')
                #current_site = get_current_site(request)
                #mail_subject = 'Activation link has been sent to your email id'
                #message = render_to_string('acc_active_email.html', {
                #    'user': user,
                #    'domain': current_site.domain,
                #    'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                #    'token': account_activation_token.make_token(user),
                #})
                #email = EmailMessage(
                #    mail_subject, message, to=[email]
                #)
                #email.send()
                #return redirect('/email-verification/')
        else:
            return render(request, 'homepage.html', {'error': 'Both passwords are not matching'})

    else:
        return render(request, 'homepage.html')


#def activate(request, uidb64, token):
#    try:
#        uid = force_str(urlsafe_base64_decode(uidb64))
#        user = User.objects.get(pk=uid)
#    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
#        user = None
#    if user is not None and account_activation_token.check_token(user, token):
#        user.is_active = True
#        user.save()
#        return redirect('/information-page/')
#    else:
#        return HttpResponse('Activation link is invalid!')
def information_page(request):
    try:  # prod = user.objects.get(pk=request.user.id)
        prod = user.objects.get(author=request.user)
        if request.method == 'POST':
            if len(request.FILES) != 0:
                if len(prod.file) > 0:
                    os.remove(prod.file.path)
                prod.file = request.FILES['file']
            prod.name = request.POST.get('name')
            prod.contact_number = request.POST.get('contact_number')
            prod.vaccination_brand = request.POST.get('vaccination_brand')
            prod.vaccination_site = request.POST.get('vaccination_site')
            prod.address = request.POST.get('address')
            prod.age = request.POST.get('age')
            prod.bday = request.POST.get('bday')
            prod.gender = request.POST.get('gender')
            prod.author = request.user
            prod.save()
            return redirect('/server-form-page/')
        return render(request, 'information.html', {'prod': prod})
    except user.DoesNotExist:
        prod = user()
        if request.method == 'POST':
            prod.name = request.POST.get('name')
            prod.contact_number = request.POST.get('contact_number')
            prod.vaccination_brand = request.POST.get('vaccination_brand')
            prod.vaccination_site = request.POST.get('vaccination_site')
            prod.address = request.POST.get('address')
            prod.age = request.POST.get('age')
            prod.bday = request.POST.get('bday')
            prod.gender = request.POST.get('gender')
            if len(request.FILES) != 0:
                prod.files = request.FILES['file']
            prod.author = request.user
            prod.save()
            return redirect('/server-form-page/')
        return render(request, 'information.html')
#def information_page(request):
#    prod = user()
#    if request.method == 'POST':
#        prod.name = request.POST.get('name')
#        prod.contact_number = request.POST.get('contact_number')
#        prod.vaccination_brand = request.POST.get('vaccination_brand')
#        prod.vaccination_site = request.POST.get('vaccination_site')
#        prod.address = request.POST.get('address')
#        prod.age = request.POST.get('age')
#        prod.bday = request.POST.get('bday')
#        prod.gender = request.POST.get('gender')
#        if len(request.FILES) != 0:
#            prod.file = request.FILES['file']
#        prod.author = request.user
#        prod.save()
#        return redirect('/server-form-page/')
#    return render(request, 'information.html')
    # prod = user()
    # if request.method == 'POST':
    #     prod.name = request.POST.get('name')
    #     prod.contact_number = request.POST.get('contact_number')
    #     prod.vaccination_brand = request.POST.get('vaccination_brand')
    #     prod.vaccination_site = request.POST.get('vaccination_site')
    #     prod.address = request.POST.get('address')
    #     prod.age = request.POST.get('age')
    #     prod.bday = request.POST.get('bday')
    #     prod.gender = request.POST.get('gender')
    #     if len(request.FILES) != 0:
    #         prod.file = request.FILES['file']
    #     prod.author = request.user.id
    #     prod.save()
    #     messages.success(request, 'Successfully Submitted')
    #     return redirect('/')
    # return render(request, 'information.html')


def profileEdit(request):
    try:  # prod = user.objects.get(pk=request.user.id)
        prod = user.objects.get(author=request.user)
        if request.method == 'POST':
            if len(request.FILES) != 0:
                if len(prod.file) > 0:
                    os.remove(prod.file.path)
                prod.file = request.FILES['file']
            prod.name = request.POST.get('name')
            prod.contact_number = request.POST.get('contact_number')
            prod.vaccination_brand = request.POST.get('vaccination_brand')
            prod.vaccination_site = request.POST.get('vaccination_site')
            prod.address = request.POST.get('address')
            prod.age = request.POST.get('age')
            prod.bday = request.POST.get('bday')
            prod.gender = request.POST.get('gender')
            prod.author = request.user
            prod.save()
            return redirect('/server-form-page/')
        return render(request, 'profileEdit.html', {'prod': prod})
    except user.DoesNotExist:
        prod = user()
        if request.method == 'POST':
            prod.name = request.POST.get('name')
            prod.contact_number = request.POST.get('contact_number')
            prod.vaccination_brand = request.POST.get('vaccination_brand')
            prod.vaccination_site = request.POST.get('vaccination_site')
            prod.address = request.POST.get('address')
            prod.age = request.POST.get('age')
            prod.bday = request.POST.get('bday')
            prod.gender = request.POST.get('gender')
            if len(request.FILES) != 0:
                prod.files = request.FILES['file']
            prod.author = request.user
            prod.save()
            return redirect('/server-form-page/')
        return render(request, 'profileEdit.html')

        # prod = user.objects.get(pk=request.user.id)
        # if request.method == 'POST':
        #     if len(request.FILES) !=0:
        #         if len(prod.file) >0:
        #             os.remove(prod.file.path)
        #         prod.file = request.FILES['file']
        #     prod.name = request.POST.get('name')
        #     prod.contact_number= request.POST.get('contact_number')
        #     prod.vaccination_brand = request.POST.get('vaccination_brand')
        #     prod.vaccination_site = request.POST.get('vaccination_site')
        #     prod.address = request.POST.get('address')
        #     prod.age = request.POST.get('age')
        #     prod.bday = request.POST.get('bday')
        #     prod.gender = request.POST.get('gender')
        #     prod.save()
        #     return redirect('/server-form-page/')
        # return render (request,'profileEdit.html',{'prod':prod})


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
        prod = sideeffect.objects.get(author=request.user)
        if request.method == 'POST':
            prod.muscle_ache = request.POST.get('muscle_ache')
            prod.headache = request.POST.get('headache')
            prod.fever = request.POST.get('fever')
            prod.redness = request.POST.get('redness')
            prod.swelling = request.POST.get('swelling')
            prod.tenderness = request.POST.get('tenderness')
            prod.warmth = request.POST.get('warmth')
            prod.itch = request.POST.get('itch')
            prod.induration = request.POST.get('induration')
            prod.feverish = request.POST.get('feverish')
            prod.chills = request.POST.get('chills')
            prod.join_pain = request.POST.get('join_pain')
            prod.fatigue = request.POST.get('fatigue')
            prod.nausea = request.POST.get('nausea')
            prod.vomiting = request.POST.get('vomiting')
            prod.author = request.user
            prod.save()
            messages.success(request, "Successfully Submitted")
            return redirect('/')
            # return render (request,'login_page.html')
        else:
            return render(request, 'sideeffect.html', {'prod': prod})
    except sideeffect.DoesNotExist:
        prod = sideeffect()
        if request.method == 'POST':
            prod.muscle_ache = request.POST.get('muscle_ache')
            prod.headache = request.POST.get('headache')
            prod.fever = request.POST.get('fever')
            prod.redness = request.POST.get('redness')
            prod.swelling = request.POST.get('swelling')
            prod.tenderness = request.POST.get('tenderness')
            prod.warmth = request.POST.get('warmth')
            prod.itch = request.POST.get('itch')
            prod.induration = request.POST.get('induration')
            prod.feverish = request.POST.get('feverish')
            prod.chills = request.POST.get('chills')
            prod.join_pain = request.POST.get('join_pain')
            prod.fatigue = request.POST.get('fatigue')
            prod.nausea = request.POST.get('nausea')
            prod.vomiting = request.POST.get('vomiting')
            prod.author = request.user
            prod.save()
            messages.success(request, "Successfully Submitted")
            return redirect('/')
        else:
            return render(request, 'sideeffect.html')


def server_form(request):
    try:
        prod = questioner.objects.get(author=request.user)
        if request.method == 'POST':
            # author_id = get
            # author = User.objects.get(id=author_id)
            prod.Q0 = request.POST.get('Q0')
            prod.Q1 = request.POST.get('Q1')
            prod.Q2 = request.POST.get('Q2')
            prod.Q3 = request.POST.get('Q3')
            prod.Q4 = request.POST.get('Q4')
            prod.Q5 = request.POST.get('Q5')
            prod.Q6 = request.POST.get('Q6')
            prod.Q7 = request.POST.get('Q7')
            prod.Q8 = request.POST.get('Q8')
            prod.Q9 = request.POST.get('Q9')
            prod.Q10 = request.POST.get('Q10')
            prod.Q11 = request.POST.get('Q11')
            prod.Q12 = request.POST.get('Q12')
            prod.Q13 = request.POST.get('Q13')
            prod.Q14 = request.POST.get('Q14')
            prod.Q15 = request.POST.get('Q15')
            prod.Q16 = request.POST.get('Q16')
            prod.Q17 = request.POST.get('Q17')
            prod.Q18 = request.POST.get('Q18')
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
            prod.author = request.user
            prod.save()
            return redirect('/success-page/')
        else:
            return render(request, 'serverform.html', {'prod': prod})

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
            prod.Q8 = request.POST.get('Q8')
            prod.Q9 = request.POST.get('Q9')
            prod.Q10 = request.POST.get('Q10')
            prod.Q11 = request.POST.get('Q11')
            prod.Q12 = request.POST.get('Q12')
            prod.Q13 = request.POST.get('Q13')
            prod.Q14 = request.POST.get('Q14')
            prod.Q15 = request.POST.get('Q15')
            prod.Q16 = request.POST.get('Q16')
            prod.Q17 = request.POST.get('Q17')
            prod.Q18 = request.POST.get('Q18')
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
            # instance = prod.save(commit=False)
            # instance.author = request.user
            # instance.save
            prod.author = request.user
            prod.save()
            return redirect('/success-page/')
        else:
            return render(request, 'serverform.html')


def success_page(request):

    template = loader.get_template('success.html')
    return HttpResponse(template.render())


def toggle_status(request, id):
    status = User.objects.get(id=id)
    status.is_active = 0
    status.save()
    return redirect('/dashboard/')


def toggle_status_active(request, id):
    status = User.objects.get(id=id)
    status.is_active = 1
    status.save()
    return redirect('/dashboard/')


# def dashboard(request):
#     item_list = user.objects.all().values().order_by('-date_created')
#     item_lists = sideeffect.objects.all().values()
#     quesT = questioner.objects.all().values()
#     userAccount = User.objects.all().values().filter(
#         is_superuser=False).order_by('-date_joined')
#     total_user = userAccount.count()
#     total_admin = User.objects.filter(is_superuser=True).count()

#     context = {
#         'item_list': item_list,
#         'item_lists': item_lists,
#         'quesT': quesT,
#         'userAccount': userAccount,
#         'total_user': total_user,
#         'total_admin': total_admin,
#     }

#     return render(request, 'system/dashboard.html', context)

def dashboard(request):
    #Total teen,adult and Senior
    childTotal = user.objects.filter(age__lte=17).values().count()
    seniorTotal = user.objects.filter(age__gte=60).values().count()
    adultTotal = user.objects.filter(age__gte=18 , age__lte=59).values().count()
    myFilter = OrderFilter()
    ###
    item_list = user.objects.all().values().order_by('-date_created')
    maleTotal = user.objects.filter(Q(gender='Male') | Q(gender='male')).values().count()
    femaleTotal = user.objects.filter(Q(gender='Female') | Q(gender='female')).values().count()
    item_lists = sideeffect.objects.all()
    item_listsTotal = sideeffect.objects.all().values().count()
    totalModerna = user.objects.filter(vaccination_brand='moderna').values().count()
    totalPfizer = user.objects.filter(vaccination_brand='pfizer').values().count()
    totalAstraZeneca = user.objects.filter(vaccination_brand='astraZeneca').values().count()
    totalSinovac = user.objects.filter(vaccination_brand='sinovac').values().count()
    totalJnj = user.objects.filter(vaccination_brand='johnson_and_Johnsons').values().count()
    quesT = questioner.objects.all().values()
    userAccount = User.objects.all().values().filter(is_superuser=False).order_by('-date_joined').order_by('-last_login')
    total_user = userAccount.count()
    total_user = userAccount.count()
    total_admin = User.objects.filter(is_superuser=True).count()
    chills = sideeffect.objects.filter(Q(chills='Yes') | Q(chills='yes')).values().count()
    fatigue = sideeffect.objects.filter(Q(fatigue='Yes') | Q(fatigue='yes')).values().count()
    feverTotal = sideeffect.objects.filter(Q(fever='Yes') | Q(fever='yes')).values().count()
    feverish = sideeffect.objects.filter(Q(feverish='Yes') | Q(feverish='yes')).values().count()
    headache = sideeffect.objects.filter(Q(headache='Yes') | Q(headache='yes')).values().count()
    induration = sideeffect.objects.filter(Q(induration='Yes') | Q(induration='yes')).values().count()
    itch = sideeffect.objects.filter(Q(itch='Yes') | Q(itch='yes')).values().count()
    join_pain = sideeffect.objects.filter(Q(join_pain='Yes') | Q(join_pain='yes')).values().count()
    muscle_ache = sideeffect.objects.filter(Q(muscle_ache='Yes') | Q(muscle_ache='yes')).values().count()
    nausea = sideeffect.objects.filter(Q(nausea='Yes') | Q(nausea='yes')).values().count()
    redness = sideeffect.objects.filter(Q(redness='Yes') | Q(redness='yes')).values().count()
    swelling = sideeffect.objects.filter(Q(swelling='Yes') | Q(swelling='yes')).values().count()
    tenderness = sideeffect.objects.filter(Q(tenderness='Yes') | Q(tenderness='yes')).values().count()
    vomiting = sideeffect.objects.filter(Q(vomiting='Yes') | Q(vomiting='yes')).values().count()
    warmth = sideeffect.objects.filter(Q(warmth='Yes') | Q(warmth='yes')).values().count()
    chillsN = sideeffect.objects.filter(Q(chills='No') | Q(chills='no')).values().count()
    fatigueN = sideeffect.objects.filter(Q(fatigue='No') | Q(fatigue='no')).values().count()
    feverTotalN = sideeffect.objects.filter(Q(fever='No') | Q(fever='no')).values().count()
    feverishN = sideeffect.objects.filter(Q(feverish='No') | Q(feverish='no')).values().count()
    headacheN = sideeffect.objects.filter(Q(headache='No') | Q(headache='no')).values().count()
    indurationN = sideeffect.objects.filter(Q(induration='No') | Q(induration='no')).values().count()
    itchN = sideeffect.objects.filter(Q(itch='No') | Q(itch='no')).values().count()
    join_painN = sideeffect.objects.filter(Q(join_pain='No') | Q(join_pain='no')).values().count()
    muscle_acheN = sideeffect.objects.filter(Q(muscle_ache='No') | Q(muscle_ache='no')).values().count()
    nauseaN = sideeffect.objects.filter(Q(nausea='No') | Q(nausea='no')).values().count()
    rednessN = sideeffect.objects.filter(Q(redness='No') | Q(redness='no')).values().count()
    swellingN = sideeffect.objects.filter(Q(swelling='No') | Q(swelling='no')).values().count()
    tendernessN = sideeffect.objects.filter(Q(tenderness='No') | Q(tenderness='no')).values().count()
    vomitingN = sideeffect.objects.filter(Q(vomiting='No') | Q(vomiting='no')).values().count()
    warmthN = sideeffect.objects.filter(Q(warmth='No') | Q(warmth='no')).values().count()

    context = {
        'childTotal': childTotal,
        'adultTotal': adultTotal,
        'seniorTotal': seniorTotal,
        'item_list': item_list,
        'item_lists': item_lists,
        'item_listsTotal': item_listsTotal,
        'quesT': quesT,
        'userAccount': userAccount,
        'total_user': total_user,
        'total_admin': total_admin,
        'maleTotal': maleTotal,
        'femaleTotal': femaleTotal,
        'totalModerna': totalModerna,
        'totalPfizer': totalPfizer,
        'totalAstraZeneca': totalAstraZeneca,
        'totalSinovac': totalSinovac,
        'totalJnj': totalJnj,
        'chills': chills,
        'fatigue': fatigue,
        'feverTotal': feverTotal,
        'feverish': feverish,
        'headache': headache,
        'induration': induration,
        'itch': itch,
        'join_pain': join_pain,
        'muscle_ache': muscle_ache,
        'nausea': nausea,
        'redness': redness,
        'swelling': swelling,
        'tenderness': tenderness,
        'vomiting': vomiting,
        'warmth': warmth,
        'chillsN':chillsN,
        'fatigueN':fatigueN,
        'feverTotalN':feverTotalN,
        'feverishN':feverishN,
        'headacheN':headacheN,
        'indurationN':indurationN,
        'itchN':itchN,
        'join_painN':join_painN,
        'muscle_acheN':muscle_acheN,
        'nauseaN':nauseaN,
        'rednessN':rednessN,
        'swellingN':swellingN,
        'tendernessN':tendernessN,
        'vomitingN':vomitingN,
        'warmthN':warmthN,
    }
    return render(request, 'home/index.html', context)


def userAccount(request):
    userAccount = User.objects.all().values().filter(
        is_superuser=False).order_by('-date_joined')
    context = {
        'userAccount': userAccount,
    }
    return render(request, 'home/UserAccount.html', context)

def informationCollection(request):
    orders = user.objects.all()
    item_list = user.objects.all().values().order_by('-date_created')
    myFilter = OrderFilter(request.GET, queryset=orders)
    orders = myFilter.qs
    context = {
        'orders': orders,
        'myFilter': myFilter,
        'item_list': item_list,
    }

    return render(request, 'home/InformationCollection.html', context)


def survey(request):
    quesT = questioner.objects.all()
    context = {
        'quesT': quesT,
    }
    return render(request, 'home/Survey.html', context)


def sideEffect(request):
    item_lists = sideeffect.objects.all()

    context = {
        'item_lists': item_lists,
    }
    return render(request, 'home/SideEffect.html', context)

# def dashboard(request):
#     item_list = user.objects.all().values().order_by('-date_created')
#     item_lists = sideeffect.objects.all().values()
#     quesT = questioner.objects.all().values()
#     userAccount = User.objects.all().values().filter(
#         is_superuser=False).order_by('-date_joined')
#     total_user = userAccount.count()
#     total_admin = User.objects.filter(is_superuser=True).count()

#     context = {
#         'item_list': item_list,
#         'item_lists': item_lists,
#         'quesT': quesT,
#         'userAccount': userAccount,
#         'total_user': total_user,
#         'total_admin': total_admin,
#     }

#     return render(request, 'layouts/LayoutDashboard.html', context)

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
