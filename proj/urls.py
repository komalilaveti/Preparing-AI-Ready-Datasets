from django.contrib import admin
from django.urls import path
from django.urls import re_path as url 
from django.conf import settings
from django.conf.urls.static import static
from app import views 

urlpatterns = [
    path('admin/', admin.site.urls),
    url('^$',views.home,name='sihpage'),
    url('index',views.index,name='Homepage'),
    url('preprocess',views.preprocess,name='preprocess'),

]  + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
 