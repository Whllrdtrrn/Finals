# Generated by Django 4.1.1 on 2022-11-23 21:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('vaccimo', '0017_alter_questioner_author_alter_sideeffect_author_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='questioner',
            name='id',
            field=models.AutoField(primary_key=True, serialize=False),
        ),
        migrations.AlterField(
            model_name='sideeffect',
            name='id',
            field=models.AutoField(primary_key=True, serialize=False),
        ),
        migrations.AlterField(
            model_name='user',
            name='id',
            field=models.AutoField(primary_key=True, serialize=False),
        ),
    ]
