from gmft import AutoTableDetector
from gmft.auto import CroppedTable, AutoTableFormatter
from gmft.pdf_bindings import PyPDFium2Document

detector = AutoTableDetector()
formatter = AutoTableFormatter()

file_path = "../../sample2/BM202404290000031873_0.pdf"


def ingest_pdf(pdf_path):  # produces list[CroppedTable]
    doc = PyPDFium2Document(pdf_path)
    tables = []
    for page in doc:
        tables += detector.extract(page)

        table = CroppedTable.from_dict(
            {
                "filename": file_path,
                "page_no": 0,
                "bbox": (
                    36.08979415893555,
                    960.8895874023438,
                    713.369873046875,
                    1373.876220703125,
                ),
                "confidence_score": 0.6,
                "label": 0,
            },
            page,
        )

        print(table.text())
    return tables, doc


tables, doc = ingest_pdf(file_path)
# print(f"tables: {tables}")
doc.close()  # once you're done with the document
