"""Import a LibimSeTi archive from a local file or a supplied HTTPS source."""

import argparse
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile


def extract_inputs(archive, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory, ZipFile(archive) as data:
        staging = Path(directory) / "raw"
        staging.mkdir()
        for name in ("ratings.dat", "gender.dat"):
            members = [entry for entry in data.infolist() if PurePosixPath(entry.filename).name == name]
            if len(members) != 1 or members[0].file_size == 0:
                raise ValueError(f"Archive must contain exactly one nonempty {name}")
            with data.open(members[0]) as source, (staging / name).open("wb") as target:
                copyfileobj(source, target)
        if output.exists():
            raise FileExistsError(f"Choose a fresh output directory: {output}")
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--url", help="HTTPS URL for an authorized copy of the original ZIP")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    if args.archive:
        extract_inputs(args.archive, args.output)
        return
    if not args.url.startswith("https://"):
        parser.error("--url must use HTTPS")
    with TemporaryDirectory() as directory:
        archive = Path(directory) / "libimseti.zip"
        with urlopen(args.url, timeout=60) as source, archive.open("wb") as target:
            copyfileobj(source, target)
        extract_inputs(archive, args.output)


if __name__ == "__main__":
    main()
