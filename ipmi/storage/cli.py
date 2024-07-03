import os
from pathlib import Path
from pprint import pprint

import click
from halo import Halo
from InquirerPy import inquirer
from tabulate import tabulate as _tabulate
from termcolor import cprint

from ipmi.storage.client import StorageClient
from ipmi.storage.dataset import Dataset, SubDataset
from ipmi.storage.errors import print_exceptions


@print_exceptions
def init_storage_client():
    return StorageClient()


storage = init_storage_client()


def tabulate(data: list[dict]) -> str:
    return _tabulate(data, maxcolwidths=64, headers="keys", tablefmt="simple_grid")


@click.group(context_settings={"show_default": True})
def cli():
    pass


@cli.command(help="Login to storage server")
@print_exceptions
def login():
    username = inquirer.text(message="Username:").execute()

    password = inquirer.secret(
        message="Password:",
        transformer=lambda _: "[hidden]",
    ).execute()
    storage.login(username=username, password=password)
    print(f"Your are now logged in as {username}")


@cli.command(help="Logout from storage server")
@print_exceptions
def logout():
    storage.logout()
    print("Your are now logged out")


@cli.group(help="Upload/download/list data sets")
def dataset():
    pass


@dataset.command(help=("Create a new data set."))
@print_exceptions
def create():
    with storage:
        dataset = storage.create_or_edit_dataset()

    if dataset:
        dataset_table = tabulate(
            [dataset.formatted_dict()],
        )
        cprint("Created the following data set:", attrs=("reverse",))
        print(dataset_table)


@dataset.command(
    help=(
        "Upload data to a data set. "
        "FOLDER has to be the root path containing all the data."
    )
)
@click.argument(
    "dataset_id",
    type=str,
)
@click.argument(
    "folder",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@print_exceptions
def upload(dataset_id: str, folder: Path):
    with storage:
        storage.upload_sub_dataset(dataset_folder=folder, dataset_id=dataset_id)


@dataset.command(help="List all available data sets")
@click.option(
    "--filter",
    type=str,
    default=None,
    help=(
        "Python string that is passed to eval(...) in order to filter the data sets. "
        "If the eval result is True the data set is returned. "
        "Note that the passed filter has to be valid Python syntax. "
        "The following packages/classes can be used: "
        "re, datetime, Modality, ProblemStatement. "
        "Example (note the single and double quotes): "
        "--filter \"dataset.name == 'dataset_42' and dataset.n_samples > 42\""
    ),
)
@click.option(
    "--sort",
    type=str,
    default="created_at",
    help=("Dataset column used for sorting (ascending)"),
)
@click.option(
    "--columns",
    type=str,
    help=(
        "Comma-separated list of columns that should be returned. "
        "By default, all columns are returned."
    ),
)
@click.option(
    "--dict",
    "dict_",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Returns the response as dict instead of formatted table. "
        "This is useful if you have limited screen space."
    ),
)
@print_exceptions
def list(filter, sort, columns, dict_):
    with storage:
        if columns:
            columns = [c.strip() for c in columns.split(",")]

        with Halo(text="Loading", spinner="dots") as spinner:

            def progress_callback(current: int, total: int):
                spinner.text = f"Loading {current}/{total}"

            datasets = storage.list_datasets(
                filter=filter, sort=sort, progress_callback=progress_callback
            )

        if datasets:
            if dict_:
                pprint(
                    [dataset.dict(exclude={"sub_datasets"}) for dataset in datasets],
                    sort_dicts=False,
                    width=os.get_terminal_size().columns,
                )
            else:
                table = tabulate(
                    [dataset.formatted_dict(columns=columns) for dataset in datasets],
                )
                print(table)
        else:
            print("No data sets were found")


@dataset.command(help="Get details of data set")
@click.argument(
    "dataset_id",
    type=str,
)
@click.option(
    "--sort",
    type=str,
    default="created_at",
    help=("Sub-dataset column used for sorting (ascending)"),
)
@click.option(
    "--columns",
    type=str,
    help=(
        "Comma-separated list of columns that should be returned. "
        "By default, all columns are returned."
    ),
)
@click.option(
    "--dict",
    "dict_",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Returns the response as dict instead of formatted table. "
        "This is useful if you have limited screen space."
    ),
)
@print_exceptions
def details(dataset_id, sort, columns, dict_):
    _details(dataset_id=dataset_id, sort=sort, columns=columns, dict_=dict_)


def _details(dataset_id, sort=None, columns=None, dict_=False):
    with storage:
        if columns:
            columns = [c.strip() for c in columns.split(",")]

        with Halo(text="Loading", spinner="dots"):
            dataset = storage.get_dataset(dataset_id)
            sub_datasets = storage.list_sub_datasets(dataset.id, sort=sort)

        if sub_datasets:
            cprint("DATASET DETAILS", attrs=("reverse",))

            if dict_:
                pprint(
                    dataset.dict(exclude={"sub_datasets"}),
                    sort_dicts=False,
                    width=os.get_terminal_size().columns,
                )
            else:
                dataset_table = tabulate(
                    [dataset.formatted_dict(columns=columns)],
                )

                print(dataset_table)
            print()

            cprint("SUB DATASET DETAILS", attrs=("reverse",))
            if dict_:
                pprint(
                    [
                        sub_dataset.dict(exclude={"sub_datasets"})
                        for sub_dataset in sub_datasets
                    ],
                    sort_dicts=False,
                    width=os.get_terminal_size().columns,
                )
            else:
                sub_dataset_table = tabulate(
                    [
                        sub_dataset.formatted_dict(columns=columns)
                        for sub_dataset in sub_datasets
                    ],
                )
                print(sub_dataset_table)
        else:
            print("No details were found")


@dataset.command(
    help=(
        "Download a data set. "
        "FOLDER has to be the root path containing all the data."
    )
)
@click.argument(
    "dataset_id",
    type=str,
)
@click.argument(
    "output_folder",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.option(
    "--use-names",
    is_flag=True,
    show_default=True,
    default=False,
    help="Name folders according to data set and sub-data set names",
)
@print_exceptions
def download(dataset_id, output_folder, use_names):
    with storage:
        dataset = storage.get_dataset(dataset_id)

        storage.download_dataset(
            dataset_id=dataset.id, output_folder=output_folder, use_names=use_names
        )


@dataset.command(
    help=(
        "Edit a (sub) data set. You can either pass the 'dataset_id' or "
        "'dataset_id/sub_dataset_id' as the identifier"
    )
)
@click.argument(
    "identifier",
    type=str,
)
# @print_exceptions
def edit(identifier):
    if "/" in identifier:
        dataset_id, sub_dataset_id = identifier.split("/")
    else:
        dataset_id = identifier
        sub_dataset_id = None

    with storage:
        with Halo(text="Loading", spinner="dots"):
            dataset = storage.get_dataset(dataset_id)

            if sub_dataset_id:
                sub_dataset = storage.get_sub_dataset(
                    sub_dataset_id, dataset_id=dataset.id
                )
                entry = sub_dataset
            else:
                entry = dataset

        # dataset is None if KeyboardInterrupt while editing
        entry = entry.edit_from_prompt()

        if entry:
            if isinstance(entry, Dataset):
                storage.create_or_edit_dataset(entry)
            elif isinstance(entry, SubDataset):
                storage.edit_sub_dataset(entry)
            else:
                raise ValueError

            table = tabulate(
                [entry.formatted_dict()],
            )
            cprint("Edited the following entry:", attrs=("reverse",))
            print(table)


@dataset.command(help="Delete a data set.")
@click.argument(
    "dataset_id",
    type=str,
)
@print_exceptions
def delete(dataset_id):
    with storage:
        with Halo(text="Loading", spinner="dots"):
            dataset = storage.get_dataset(dataset_id)
            sub_datasets = storage.list_sub_datasets(dataset.id)

        _details(dataset_id)

        if inquirer.confirm(
            message=(
                f"Do you really want to delete the following "
                f"dataset containing {len(sub_datasets)} sub-data set(s)?"
            )
        ).execute():
            with Halo(text="Deleting", spinner="dots"):
                storage.delete_dataset(dataset_id=dataset.id)
