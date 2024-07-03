import uuid
from datetime import datetime

from peewee import (
    BlobField,
    BooleanField,
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


class BaseModel(Model):
    class Meta:
        database = database


class Modality(BaseModel):
    name = CharField(primary_key=True)


class Anatomy(BaseModel):
    name = CharField(primary_key=True)


class Dataset(BaseModel):
    name = CharField(primary_key=True)


class Metric(BaseModel):
    name = CharField(primary_key=True)
    lower_is_better = BooleanField()


class Image(BaseModel):
    uuid = UUIDField(primary_key=True, default=uuid.uuid4)
    name = CharField(max_length=255)
    modality = ForeignKeyField(Modality, backref="images", on_delete="CASCADE")
    anatomy = ForeignKeyField(Anatomy, backref="images", on_delete="CASCADE")
    dataset = ForeignKeyField(Dataset, backref="images", on_delete="CASCADE")

    class Meta:
        indexes = (
            # unique index
            (("name", "dataset"), True),
        )


class ImagePairFeature(BaseModel):
    uuid = UUIDField(primary_key=True, default=uuid.uuid4)
    moving_image = ForeignKeyField(
        Image, backref="image_pair_features", on_delete="CASCADE"
    )
    fixed_image = ForeignKeyField(
        Image, backref="image_pair_features", on_delete="CASCADE"
    )
    feature_name = CharField(max_length=255)
    feature = BlobField()

    class Meta:
        indexes = (
            # unique index
            (("moving_image", "fixed_image", "feature_name"), True),
        )


class Run(BaseModel):
    uuid = UUIDField(primary_key=True, default=uuid.uuid4)
    moving_image = ForeignKeyField(Image, backref="runs", on_delete="CASCADE")
    fixed_image = ForeignKeyField(Image, backref="runs", on_delete="CASCADE")
    parameters = JSONField()

    created = DateTimeField(default=datetime.now)


class RunMetrics(BaseModel):
    uuid = UUIDField(primary_key=True, default=uuid.uuid4)
    run = ForeignKeyField(Run, backref="run_metrics", on_delete="CASCADE")

    metric = ForeignKeyField(Metric, backref="run_metrics", on_delete="CASCADE")
    value_before = FloatField()
    value_after = FloatField()

    class Meta:
        indexes = (
            # unique index
            (("run", "metric"), True),
        )
