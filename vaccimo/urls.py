from django.urls import path
from django.contrib import admin
from vaccimo import views

urlpatterns = [
    #     path('admin/', admin.site.urls),
    path('', views.LayoutHome, name='LayoutHome'),
    #     path('admin/', views.homepage, name='homepage'),
    #     path('index/', views.LayoutIndex, name='LayoutIndex'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('dashboard/user-account/', views.userAccount, name='userAccount'),
    path('dashboard/information-collection/',
         views.informationCollection, name='informationCollection'),
    path('dashboard/survey/', views.survey, name='survey'),
    path('dashboard/side-effect/', views.sideEffect, name='sideEffect'),

    path('', views.home_page, name='homepage'),
    path('services/', views.services, name='services'),
    path('contacts/', views.contacts, name='contacts'),
    path('aboutUs/', views.aboutUs, name='aboutUs'),
    path('loginpage/', views.login_page, name='loginpage'),
    path('register/', views.register_page, name='register'),
    path('view-details/', views.viewDetails, name='viewDetails'),
    path('information-page/', views.information_page, name='information'),
    path('Profile/', views.profileEdit, name='profileEdit'),
    path('server-form-page/', views.server_form, name='server'),
    path('side-effect-page/', views.sideeffect_page, name='sideeffect'),
    path('success-page/', views.success_page, name='success'),
    #     path('dashboard/', views.dashboard, name='dashboard'),
    path('toggle_status/<str:id>', views.toggle_status, name='toggle_status'),
    path('toggle_status_active/<str:id>',
         views.toggle_status_active, name='toggle_status_active'),
    path('logout/', views.logout_view, name='logout'),
    path('preprocessing/', views.preprocessing, name='preprocessing'),
    path('checker_page/', views.checker_page, name='checker_page'),
    path('chooseMethod/', views.chooseMethod, name='chooseMethod'),
    # path('classification/', views.classification, name='classification'),
    path('clustering/', views.clustering, name='clustering'),
#    path('activate/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20})/',
#         views.activate, name='activate'),
#    path('email-verification/', views.verification, name='verification'),

    # path('delete/', views.delete, name='delete'),
    # path('edit/<str:pk>', views.edit, name='edit') ,

]
