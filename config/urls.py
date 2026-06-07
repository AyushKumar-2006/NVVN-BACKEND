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
from power.api_tft import router as tft_router
from power import views_cg




api = NinjaAPI()

api.add_router("/power/", power_router)
api.add_router("/power/", compare_router)
api.add_router("/power/", tft_router)

urlpatterns = [
    path("", TemplateView.as_view(template_name="power/forecast.html"), name="home"),
    path("dashboard/", TemplateView.as_view(template_name="power/dashboard.html"), name="dashboard"),

    # --- CG (Chhattisgarh) forecast: dashboard + JSON API (plain Django views) ---
    # Listed before the django-ninja `api/` include so these exact paths win.
    path("dashboard/cg/", views_cg.dashboard, name="cg_dashboard"),
    path("dashboard/cg/intraday/", views_cg.dashboard_intraday, name="cg_dash_intraday"),
    path("dashboard/cg/forecast/", views_cg.dashboard_forecast, name="cg_dash_forecast"),
    path("dashboard/cg/temperature/", views_cg.dashboard_temperature, name="cg_dash_temperature"),
    path("dashboard/cg/energy/", views_cg.dashboard_energy, name="cg_dash_energy"),
    path("api/cg/forecast/", views_cg.forecast, name="cg_forecast"),
    path("api/cg/actuals/", views_cg.actuals, name="cg_actuals"),
    path("api/cg/compare/", views_cg.compare, name="cg_compare"),
    path("api/cg/model-stats/", views_cg.model_stats, name="cg_model_stats"),
    path("api/cg/intraday/", views_cg.intraday, name="cg_intraday"),
    path("api/cg/weather/", views_cg.weather, name="cg_weather"),
    path("api/cg/districts-temp/", views_cg.districts_temp, name="cg_districts_temp"),
    path("api/cg/energy-trend/", views_cg.energy_trend, name="cg_energy_trend"),

    path('admin/', admin.site.urls),
    path("api/", api.urls),
]



if settings.DEBUG:
 urlpatterns += static( settings.STATIC_URL, document_root=settings.STATIC_ROOT)