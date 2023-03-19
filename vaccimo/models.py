from datetime import datetime
from distutils.command.upload import upload
from django.db import models
from django.contrib.auth.models import User

import datetime
import os
# Create your models here.


def filepath(request, filename):
    old_filename = filename
    timeNow = datetime.datetime.now().strftime('%y%m%d%H:%M:%S')
    filename = "%s%s" % (timeNow, old_filename)
    return os.path.join('uploads/', filename)


class user(models.Model):
    id = models.AutoField(primary_key=True,)
    file = models.ImageField(upload_to=filepath, null=True, blank=True)
    email = models.CharField(max_length=100, null=True)
    name = models.CharField(max_length=100, null=True)
    contact_number = models.CharField(max_length=100, null=True)
    vaccination_brand = models.CharField(max_length=100, null=True)
    vaccination_site = models.CharField(max_length=100, null=True)
    address = models.CharField(max_length=100, null=True)
    age = models.CharField(max_length=100, null=True)
    bday = models.CharField(max_length=100, null=True)
    gender = models.CharField(max_length=100, null=True)
    date_created = models.DateField(auto_now_add=True, null=True)
    author = models.ForeignKey(
        User, default=None, on_delete=models.CASCADE, null=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "user"


class sideeffect (models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, null=True)
    muscle_ache = models.CharField(max_length=100, null=True)
    headache = models.CharField(max_length=100, null=True)
    fever = models.CharField(max_length=100, null=True)
    redness = models.CharField(max_length=100, null=True)
    swelling = models.CharField(max_length=100, null=True)
    tenderness = models.CharField(max_length=100, null=True)
    warmth = models.CharField(max_length=100, null=True)
    itch = models.CharField(max_length=100, null=True)
    induration = models.CharField(max_length=100, null=True)
    feverish = models.CharField(max_length=100, null=True)
    chills = models.CharField(max_length=100, null=True)
    join_pain = models.CharField(max_length=100, null=True)
    fatigue = models.CharField(max_length=100, null=True)
    nausea = models.CharField(max_length=100, null=True)
    vomiting = models.CharField(max_length=100, null=True)
    author = models.ForeignKey(
        User, default=None, on_delete=models.CASCADE, null=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "sideeffect"


class questioner (models.Model):

    id = models.AutoField(primary_key=True)
    Q0 = models.CharField(max_length=100, null=True)
    Q1 = models.CharField(max_length=100, null=True)
    Q2 = models.CharField(max_length=100, null=True)
    Q3 = models.CharField(max_length=100, null=True)
    Q4 = models.CharField(max_length=100, null=True)
    Q5 = models.CharField(max_length=100, null=True)
    Q6 = models.CharField(max_length=100, null=True)
    Q7 = models.CharField(max_length=100, null=True)
    Q8 = models.CharField(max_length=100, null=True)
    Q9 = models.CharField(max_length=100, null=True)
    Q10 = models.CharField(max_length=100, null=True)
    Q11 = models.CharField(max_length=100, null=True)
    Q12 = models.CharField(max_length=100, null=True)
    Q13 = models.CharField(max_length=100, null=True)
    Q14 = models.CharField(max_length=100, null=True)
    Q15 = models.CharField(max_length=100, null=True)
    Q16 = models.CharField(max_length=100, null=True)
    Q17 = models.CharField(max_length=100, null=True)
    Q18 = models.CharField(max_length=100, null=True)
    Q19 = models.CharField(max_length=100, null=True)
    Q20 = models.CharField(max_length=100, null=True)
    Q21 = models.CharField(max_length=100, null=True)
    Q22 = models.CharField(max_length=100, null=True)
    allergy = models.CharField(max_length=100, null=True)
    allergy1 = models.CharField(max_length=100, null=True)
    allergy2 = models.CharField(max_length=100, null=True)
    allergy3 = models.CharField(max_length=100, null=True)
    allergy4 = models.CharField(max_length=100, null=True)
    allergy5 = models.CharField(max_length=100, null=True)
    Q23 = models.CharField(max_length=100, null=True)
    Q24 = models.CharField(max_length=100, null=True)
    author = models.ForeignKey(
        User, default=None, on_delete=models.CASCADE, null=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "questioner"

# class User(AbstractUser):
#     is_admin = models.BooleanField(default=False)
#     is_customer = models.BooleanField(default=False)
