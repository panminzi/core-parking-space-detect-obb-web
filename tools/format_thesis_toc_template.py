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


def set_toc_title(para: ET.Element) -> None:
    ppr = para.find("w:pPr", NS)
    if ppr is None:
        ppr = ET.Element(w("pPr"))
        para.insert(0, ppr)
    jc = ppr.find("w:jc", NS)
    if jc is None:
        jc = ET.SubElement(ppr, w("jc"))
    jc.set(w("val"), "center")


def infer_level(text: str) -> int:
    first = text.split()[0] if text.split() else ""
    if first.count(".") >= 2:
        return 3
    if first.count(".") == 1:
        return 2
    return 1


def split_title_page(text: str) -> tuple[str, str]:
    page = ""
    while text and text[-1].isdigit():
        page = text[-1] + page
        text = text[:-1]
    return text.strip(), page


def format_toc_item(para: ET.Element, original_text: str) -> None:
    title, page = split_title_page(original_text)
    level = infer_level(title)

    for child in list(para):
        para.remove(child)

    ppr = ET.SubElement(para, w("pPr"))
    spacing = ET.SubElement(ppr, w("spacing"))
    spacing.set(w("line"), "360")
    spacing.set(w("lineRule"), "auto")
    ind = ET.SubElement(ppr, w("ind"))
    ind.set(w("left"), str({1: 0, 2: 360, 3: 720}.get(level, 0)))
    tabs = ET.SubElement(ppr, w("tabs"))
    tab = ET.SubElement(tabs, w("tab"))
    tab.set(w("val"), "right")
    tab.set(w("leader"), "dot")
    tab.set(w("pos"), "9000")
    jc = ET.SubElement(ppr, w("jc"))
    jc.set(w("val"), "left")

    run_title = ET.SubElement(para, w("r"))
    text_title = ET.SubElement(run_title, w("t"))
    text_title.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_title.text = title

    run_tab = ET.SubElement(para, w("r"))
    ET.SubElement(run_tab, w("tab"))

    run_page = ET.SubElement(para, w("r"))
    text_page = ET.SubElement(run_page, w("t"))
    text_page.text = page


def main() -> None:
    target = Path(r"C:\Users\hhhh\Desktop\潘敏姿的论文_图文规范版.docx")
    temp = target.with_suffix(".toc-template.docx")
    rewritten = target.with_suffix(".toc-template.rewritten.docx")
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

        set_toc_title(children[toc_idx])
        for child in children[toc_idx + 1:body_start]:
            text = text_of(child)
            if child.tag == w("p") and text:
                format_toc_item(child, text)

        new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        with zipfile.ZipFile(rewritten, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data = new_xml
                zout.writestr(item, data)

    shutil.move(str(rewritten), str(target))
    temp.unlink(missing_ok=True)
    print(f"formatted={target}")


if __name__ == "__main__":
    main()
