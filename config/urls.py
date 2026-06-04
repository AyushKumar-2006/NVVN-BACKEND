"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView
from ninja import NinjaAPI
from power.api import router as power_router
from power.api_compare import router as compare_router




api = NinjaAPI()

api.add_router("/power/", power_router)
api.add_router("/power/", compare_router)

urlpatterns = [
    path("", TemplateView.as_view(template_name="power/forecast.html"), name="home"),
    path("dashboard/", TemplateView.as_view(template_name="power/dashboard.html"), name="dashboard"),
    path('admin/', admin.site.urls),
    path("api/", api.urls),
]



if settings.DEBUG:
 urlpatterns += static( settings.STATIC_URL, document_root=settings.STATIC_ROOT)