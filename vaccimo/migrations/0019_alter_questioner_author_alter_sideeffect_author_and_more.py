# Generated by Django 4.1.1 on 2023-04-03 09:15

import datetime
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('vaccimo', '0018_alter_questioner_id_alter_sideeffect_id_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='questioner',
            name='author',
            field=models.ForeignKey(default=datetime.date, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='sideeffect',
            name='author',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='user',
            name='author',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.CreateModel(
            name='sideeffectYes',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('btnYes', models.CharField(max_length=100, null=True, unique=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('author_sideeffect', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='vaccimo.sideeffect')),
            ],
            options={
                'db_table': 'sideeffectYes',
            },
        ),
    ]