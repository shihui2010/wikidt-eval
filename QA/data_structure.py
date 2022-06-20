import os
import json
import pandas as pd
from typing import List, Union, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class QAExample:
    subset: str
    sample_id: str
    table_id: str
    question: str
    answer: List[str]

    all_table_files_web: List[str] = field(default_factory=list)
    all_ocr_files_web: List[str] = field(default_factory=list)
    table_retrieval_file_web: Union[str, None] = None
    ocr_retrieval_file_web: Union[Tuple[int, str], None] = None
        
    all_table_files_textract: List[str] = field(default_factory=list)
    all_ocr_files_textract: List[str] = field(default_factory=list)
    table_retrieval_file_textract: Union[str, None] = None
    ocr_retrieval_file_textract: Union[Tuple[int, str], None] = None



@dataclass
class OCRExample:
    text: str
    bbox: List[int]


def read_table(data_dir, fname) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, fname), sep='\t',
                     index_col=0).astype(str)
    df = df.applymap(lambda x: "" if x == "<js button>" else x)
    df = df.applymap(lambda x: "<TLDR>" if len(x) > 100 else x)
    return df


def read_ocr(data_dir, fname) -> List[OCRExample]:
    with open(os.path.join(data_dir, fname)) as fp:
        ocr_data = json.load(fp)
    res = dict()
    for idx, ocrs in ocr_data:
        this_table = list()
        res[idx] = this_table
        for text, bbox in ocrs:
            this_table.append(OCRExample(text, bbox))
    return res


class BaseDataReader:
    def __init__(self, data_dir: str, split: str, source: str):
        self.qa_examples = list()
        self.data_dir = data_dir
        with open(f"{data_dir}/samples/{split}.json") as fp:
            for line in fp:
                self.qa_examples.append(QAExample(**json.loads(line)))

    def build_table_index(self, source):
        """ Table index are sub-page independent"""
        assert source in ["web", "textract"], f"Unknown source: {source}"
        invalid_samples = []
        table_index = dict()   # table_id : tables
        attr = "table_retrieval_file_" + source
        for sid, sample in enumerate(self.qa_examples):
            table_file = sample.__getattribute__(attr)
            if table_file is None:
                invalid_samples.append(sid)
                continue
            if table_file in table_index:
                continue
            table_index[table_file] = read_table(self.data_dir, table_file)
        print(f"removing {len(invalid_samples)} from index due to empty table")
        for i in invalid_samples[::-1]:
            self.qa_examples.pop(i)
        return table_index

    def build_ocr_index(self, source):
        """ OCR index are per sub-page, e.g. all OCRs from the same
        subpage are stored in one file"""
        assert source in ["web", "textract"], f"Unknown source: {source}"
        ocr_index = dict()   # {seg_page: {table_id: list_of_ocrs}}
        attr = "ocr_retrieval_file_" + source
        for sample in self.qa_examples:
            table_id, ocr_file = sample.__getattribute__(attr)
            if ocr_file in ocr_index:
                continue
            ocr_index[ocr_file] = read_ocr(self.data_dir, ocr_file)
        return ocr_index


class TableQADataset(BaseDataReader):
    def __init__(self, data_dir, split, source):
        super(TableQADataset, self).__init__(data_dir, split, source)
        self.table_index = self.build_table_index(source)


class VQADataset(BaseDataReader):
    def __init__(self, data_dir, split, source):
        super(VQADataset, self).__init__(data_dir, split, source)
        self.ocr_index = self.build_table_index(source)
