import uuid

from peewee import (
    CharField,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    Model,
    SqliteDatabase,
    UUIDField,
)

from vroc.hyperopt_database.fields import JSONField

database = SqliteDatabase(None)

from datetime import datetime


class BaseModel(Model):
    class Meta:
        database = database


class Image(BaseModel):
    id = CharField(primary_key=True)


class Run(BaseModel):
    uuid = UUIDField(primary_key=True, default=uuid.uuid4)
    image = ForeignKeyField(Image, backref="runs", on_delete="CASCADE")
    parameters = JSONField()

    # performance
    metric_before = FloatField()
    metric_after = FloatField()

    level_metrics = JSONField()

    created = DateTimeField(default=datetime.now)
