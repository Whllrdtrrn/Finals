# Generated by Django 4.1.1 on 2022-11-23 15:57

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('vaccimo', '0012_questioner_allergy1_questioner_allergy2_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='questioner',
            name='author',
            field=models.ForeignKey(default=None, null=True, on_delete=django.db.models.deletion.CASCADE, to='vaccimo.sideeffect'),
        ),
        migrations.AddField(
            model_name='sideeffect',
            name='author',
            field=models.ForeignKey(default=None, null=True, on_delete=django.db.models.deletion.CASCADE, to='vaccimo.user'),
        ),
        migrations.AddField(
            model_name='user',
            name='author',
            field=models.ForeignKey(default=None, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]
