from django.urls import path
from django.contrib import admin  

from vaccimo import views


urlpatterns = [
    path('admin/', admin.site.urls),  
    path('', views.home_page, name='homepage' ),
    path('login-page/', views.login_page, name='loginpage' ),
    path('register-page/', views.register_page, name='register' ),
    path('view-details/', views.viewDetails, name='viewDetails' ),
    path('information-page/', views.information_page, name='information' ),
    path('server-form-page/', views.server_form, name='server' ),
    path('side-effect-page/', views.sideeffect_page, name='sideeffect' ),
    path('success-page/', views.success_page, name='success' ),
    path('dashboard/', views.dashboard, name='dashboard' ),
    path('/',views.logout_view,name='logout' ),
    # path('delete/', views.delete, name='delete'),
    # path('edit/<str:pk>', views.edit, name='edit') ,  

]