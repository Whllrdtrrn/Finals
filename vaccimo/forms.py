from django import forms
from vaccimo.models import user
# from vaccimo.models import User
from vaccimo.models import sideeffect
from vaccimo.models import questioner
from vaccimo.models import sideeffectYes

from django.forms import NumberInput, DateInput
from bootstrap_datepicker_plus.widgets import DatePickerInput, TimePickerInput, DateTimePickerInput, MonthPickerInput, YearPickerInput



# creating a form
# class UserForm(forms.ModelForm):

#     # create meta class
#     class Meta:
#         # specify model to be used
#         model = User

#         # specify fields to be used
#         fields = [
#            'username',"email",'password','confirm_password','is_admin','is_customer'
#         ]

class userForm(forms.ModelForm):
    # create meta class
    class Meta:
        # specify model to be used
        model = user

        # specify fields to be used
        fields = [
            'file', 'email', 'password', 'name', 'contact_number', 'vaccination_brand',
            'vaccination_site', 'address', 'age', 'gender', 'contact_number','date_created'
        ]
        
class sideeffectYesForm(forms.ModelForm):
    # create meta class
    class Meta:
        # specify model to be used
        model = sideeffectYes

        # specify fields to be used
        fields = [ 'btnYes' ]
        

class sideeffectForm(forms.ModelForm):
    # create meta class
    class Meta:
        # specify model to be used
        model = sideeffect

        # specify fields to be used
        fields = [
            'muscle_ache', 'headache', 'fever', 'redness', 'swelling', 'tenderness', 'warmth', 'itch', 'induration',
            'feverish', 'chills', 'join_pain', 'fatigue', 'nausea', 'vomiting', 'name'
        ]


class questionerForm(forms.ModelForm):
    # create meta class
    class Meta:
        # specify model to be used
        model = questioner

        # specify fields to be used
        fields = [
            'Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7',
            'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15',
            'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q21', 'Q22', 'allergy', 'allergy1', 'allergy2', 'allergy3', 'allergy4', 'allergy5', 'Q23', 'Q24',
        ]

class questionerForm(forms.ModelForm):
    # create meta class
    class Meta:
        # specify model to be used
        model = questioner

        # specify fields to be used
        fields = [
            'Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7',
            'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15',
            'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q21', 'Q22', 'allergy', 'allergy1', 'allergy2', 'allergy3', 'allergy4', 'allergy5', 'Q23', 'Q24',
        ]