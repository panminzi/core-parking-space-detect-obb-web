from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W_NS)
NS = {"w": W_NS}


def w(name: str) -> str:
    return f"{{{W_NS}}}{name}"


def text_of(elem: ET.Element) -> str:
    return "".join(t.text or "" for t in elem.findall(".//w:t", NS)).strip()


def set_center(para: ET.Element) -> None:
    ppr = para.find("w:pPr", NS)
    if ppr is None:
        ppr = ET.Element(w("pPr"))
        para.insert(0, ppr)
    for tabs in ppr.findall("w:tabs", NS):
        ppr.remove(tabs)
    jc = ppr.find("w:jc", NS)
    if jc is None:
        jc = ET.SubElement(ppr, w("jc"))
    jc.set(w("val"), "center")


def main() -> None:
    target = Path(r"C:\Users\hhhh\Desktop\潘敏姿的论文_图文规范版.docx")
    temp = target.with_suffix(".toc.docx")
    rewritten = target.with_suffix(".toc.rewritten.docx")
    shutil.copy2(target, temp)

    with zipfile.ZipFile(temp, "r") as zin:
        root = ET.fromstring(zin.read("word/document.xml"))
        body = root.find("w:body", NS)
        if body is None:
            raise RuntimeError("document body not found")
        children = list(body)
        toc_idx = next(i for i, child in enumerate(children) if text_of(child).replace(" ", "") == "目录")
        body_start = next(
            i for i, child in enumerate(children[toc_idx + 1:], toc_idx + 1)
            if text_of(child) == "1 绪论"
        )

        for child in children[toc_idx:body_start]:
            if child.tag == w("p"):
                set_center(child)

        new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        with zipfile.ZipFile(rewritten, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data = new_xml
                zout.writestr(item, data)

    shutil.move(str(rewritten), str(target))
    temp.unlink(missing_ok=True)
    print(f"centered={target}")


if __name__ == "__main__":
    main()
