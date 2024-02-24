import os
import tarfile
from torchdata.datapipes.iter import TarArchiveLoader
from typing import cast, IO, Iterable, Iterator, Optional, Tuple, Dict
from torchdata.datapipes import functional_datapipe
from io import BufferedIOBase
from torchdata.datapipes.utils import StreamWrapper
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple
import warnings
import torchdata.datapipes as dp
from torchdata.datapipes.iter import IterDataPipe
import json


@functional_datapipe("load_from_tar_wo_exception")
class TarArchiveLoaderWoException(TarArchiveLoader):
    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                if isinstance(data_stream, StreamWrapper) and isinstance(data_stream.file_obj, tarfile.TarFile):
                    tar = data_stream.file_obj
                else:
                    reading_mode = (self.mode if hasattr(data_stream, "seekable") and data_stream.seekable() else
                                    self.mode.replace(":", "|"))
                    # typing.cast is used here to silence mypy's type checker
                    tar = tarfile.open(fileobj=cast(Optional[IO[bytes]], data_stream), mode=reading_mode)
                for tarinfo in tar:
                    if not tarinfo.isfile():
                        continue
                    extracted_fobj = tar.extractfile(tarinfo)
                    if extracted_fobj is None:
                        warnings.warn(f"failed to extract file {tarinfo.name} from source tarfile {pathname}")
                        raise tarfile.ExtractError
                    inner_pathname = os.path.normpath(os.path.join(pathname, tarinfo.name))
                    yield inner_pathname, StreamWrapper(extracted_fobj, data_stream, name=inner_pathname)  # type: ignore[misc]
            except Exception as e:
                warnings.warn(f"Unable to extract files from corrupted tarfile stream {pathname} due to: {e}, abort!")
                # raise e
            finally:
                if isinstance(data_stream, StreamWrapper):
                    data_stream.autoclose()


@dp.functional_datapipe("parse_coco_annotations")
class CocoAnnParser(dp.iter.IterDataPipe):
    def __init__(self, source_datapipe) -> None:
        # if isinstance(source_datapipe, dict) and 'annotations' in source_datapipe:
        #     assert isinstance(source_datapipe['annotations'], list)
        #     self.source_datapipe = source_datapipe['annotations']
        # elif isinstance(source_datapipe, list):
        #     self.source_datapipe = source_datapipe
        # else:
        #     raise TypeError(f'Unexpected type of coco file: {type(source_datapipe)}')
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for _, json_data in self.source_datapipe:
            if isinstance(json_data, dict) and 'annotations' in json_data:
                assert isinstance(json_data['annotations'], list)
                iter_data = json_data['annotations']
            elif isinstance(json_data, list):
                iter_data = json_data
            else:
                raise TypeError(f'Unexpected type of coco file: {type(json_data)}')
            for item in iter_data:
                if 'caption' in item and 'image' in item:
                    yield item


@functional_datapipe("parse_jsonl_files")
class JsonlParserIterDataPipe(IterDataPipe[Tuple[str, Dict]]):
    def __init__(self, source_datapipe: IterDataPipe[Tuple[str, IO]], **kwargs) -> None:
        self.source_datapipe: IterDataPipe[Tuple[str, IO]] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[Tuple[str, Dict]]:
        for file_name, stream in self.source_datapipe:
            for idx, line in enumerate(stream):
                if line.strip() != '':
                    try:
                        yield f'{file_name}_line{idx}', json.loads(line)
                    except Exception as e:
                        warnings.warn(f"Error occured when parsing string to json due to: {e} abort!")
