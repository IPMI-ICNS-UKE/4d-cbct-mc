import fnmatch
import logging
import time
from datetime import timedelta
from functools import partial
from pathlib import Path

import click
from ct import create_ct_segmentations
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ipmi.common.logger import LoggerMixin, tqdm

logger = logging.getLogger(__name__)


class NewFileHandler(FileSystemEventHandler, LoggerMixin):
    def __init__(
        self, file_pattern: str, action_callback: callable, filenames_set: set
    ):
        super().__init__()
        self.file_pattern = file_pattern
        self.action_callback = action_callback
        self.filenames_set = filenames_set

    def on_created(self, event):
        if event.is_directory:
            return

        file_name = Path(event.src_path).name
        self._process_file_event(file_name, event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return

        file_name = Path(event.dest_path).name
        self._process_file_event(file_name, event.dest_path)

    def _process_file_event(self, file_name, file_path):
        click.echo(f"Event detected: {file_path}")
        if fnmatch.fnmatch(file_name, f"{self.file_pattern}"):
            if file_path not in self.filenames_set:
                self.logger.info(
                    f"New file added matching the pattern {self.file_pattern}: "
                    f"{file_path}"
                )
                self.filenames_set.add(file_path)
                self.action_callback(file_path)


class DirectoryMonitor:
    def __init__(
        self,
        directory: Path,
        file_pattern: str,
        action_callback: callable,
        recursive: bool = True,
    ):
        self.directory = directory
        self.file_pattern = file_pattern
        self.action_callback = action_callback
        self.filenames_set = set()
        self.observer = Observer()
        self.recursive = recursive

    def __enter__(self):
        self._scan_existing_files()
        self.event_handler = NewFileHandler(
            self.file_pattern, self.action_callback, self.filenames_set
        )
        self.observer.schedule(self.event_handler, str(self.directory), recursive=True)
        self.observer.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.observer.stop()
        self.observer.join()

    def _scan_existing_files(self):
        if self.recursive:
            files = self.directory.rglob(f"{self.file_pattern}")
        else:
            files = self.directory.glob(f"{self.file_pattern}")
        filtered_files = [
            file_path
            for file_path in files
            if file_path.is_file() and str(file_path) not in self.filenames_set
        ]
        lb = "\n"
        click.echo(
            f"Initially found files: {lb} { lb.join([str(f) for f in filtered_files])}"
        )
        for file_path in tqdm(filtered_files):
            self.filenames_set.add(str(file_path))
            self.action_callback(file_path)


def create_segmentations_for_img(
    img_filepath: str,
    force_segmentation: bool,
    segmentations: tuple[str],
    gpu_id: int,
):
    img_filepath = Path(img_filepath)
    output_dir = img_filepath.parent / f"segmentations_{img_filepath.stem}"
    if output_dir.is_dir() and not force_segmentation:
        click.echo(
            f"{output_dir=} already exists, assuming segmentations exist."
            f"Skipped...."
        )
        return
    output_dir.mkdir(exist_ok=True)
    try:
        click.echo(
            f"Segmentation of {img_filepath} is started, saving to {output_dir}..."
        )
        start = time.time()
        create_ct_segmentations(
            image_filepath=img_filepath,
            output_folder=output_dir,
            models=list(segmentations),
            gpu_id=gpu_id,
        )
        duration = time.time() - start
        click.echo(
            f"Segmentation of {img_filepath} is completed and took "
            f"{str(timedelta(seconds=duration))} h:min:sec."
        )
    except Exception as e:
        click.echo(f"Error: {e} for {img_filepath=}")


@click.command()
@click.option(
    "-d",
    "--search-dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    help="Directory containing CT images",
)
@click.option(
    "-p",
    "--img-pattern",
    required=True,
    type=click.STRING,
    show_default=True,
    default="*.nii",
    help="Image filename pattern to search for",
)
@click.option(
    "-f",
    "--force-segmentation",
    is_flag=True,
    required=False,
    type=click.BOOL,
    show_default=True,
    default=False,
    help="Force segmentation even if output dir exists",
)
@click.option(
    "-s",
    "--segmentations",
    required=False,
    multiple=True,
    type=click.Choice(("total", "body", "lung_vessels")),
    show_default=True,
    default=("total", "body", "lung_vessels"),
    help="Segmentations to create",
)
@click.option(
    "-r",
    "--search-recursively",
    is_flag=True,
    required=False,
    type=click.BOOL,
    show_default=True,
    default=True,
    help="Search for images recursively in search-dir",
)
@click.option(
    "-g",
    "--gpu-id",
    required=False,
    type=click.INT,
    show_default=True,
    default=0,
    help="GPU ID to use for segmentation",
)
def create_segmentations(
    search_dir: Path,
    img_pattern: str,
    force_segmentation: bool,
    segmentations: list[str],
    search_recursively,
    gpu_id,
):
    perform_segmentation = partial(
        create_segmentations_for_img,
        force_segmentation=force_segmentation,
        segmentations=segmentations,
        gpu_id=gpu_id,
    )

    with DirectoryMonitor(
        search_dir, img_pattern, perform_segmentation, search_recursively
    ):
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    create_segmentations()
