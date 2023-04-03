import django_filters
from django_filters import DateFilter
from .models import *
from django.forms import DateInput
from django import forms

## information collection
class OrderFilter(django_filters.FilterSet):
    start_date = DateFilter(field_name="date_created", lookup_expr='gte')

    class Meta:
        model = user
        fields = ['name', 'vaccination_brand','age','gender','date_created']
        exclude = ['author','date_created']
 
## side effect       
class EffectFilter(django_filters.FilterSet):
    start_date = DateFilter(field_name="date_created", lookup_expr='gte')

    class Meta:
        model = sideeffect
        fields = ['name', 'muscle_ache','headache','fever','redness','swelling','tenderness','warmth','itch','induration','feverish','chills','join_pain','fatigue','nausea','vomiting','redness']
        exclude = ['author','date_created']      