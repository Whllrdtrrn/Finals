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







# Create your views here.
def home_page(request):
    return render (request,'home_page.html')


def logout_view(request):
    logout(request)
    return redirect('login-page/')


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
                return redirect('/information-page/')
                # return HttpResponse('/information-page/')          
                # return redirect('/information-page/')
         else:
            return render (request,'login_page.html', {'error':'Invalid Username or Password'}) 
     else:
         return render(request,'login_page.html')
    

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
                return redirect(register_page)
            elif User.objects.filter(email=email).exists():
                messages.info(request, 'Email is already taken')
                return redirect(register_page)
            else:
                user = User.objects.create_user(username=username, password=password, 
                email=email,first_name=first_name,last_name=last_name)
                user.save()
                
                return redirect('loginpage')

        else:
            messages.info(request, 'Both passwords are not matching')
            return redirect('register_page')
            
    else:
        return render(request, 'register_page.html')  

def information_page (request):
    try:
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
        return render (request,'information.html',{'prod':prod})
    except user.DoesNotExist:
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
                return redirect('/server-form-page/')
        return render (request,'information.html')    
            
    

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
                return redirect('/success-page/')
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
                return redirect('/success-page/')
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


def dashboard (request):
     item_list = user.objects.all().values()
     item_lists = sideeffect.objects.all().values()
     quesT = questioner.objects.all().values()
     userAccount = User.objects.all().values().filter(is_superuser=False).filter(is_active=True)
     total_user = userAccount.count()
     total_admin = User.objects.filter(is_superuser=True).count()
     context = {
        'item_list': item_list,
        'item_lists': item_lists,
        'quesT': quesT,
        'userAccount': userAccount,
        'total_user':total_user,
        'total_admin':total_admin
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