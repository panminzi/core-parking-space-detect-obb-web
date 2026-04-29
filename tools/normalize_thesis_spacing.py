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


def ppr_of(para: ET.Element) -> ET.Element:
    ppr = para.find("w:pPr", NS)
    if ppr is None:
        ppr = ET.Element(w("pPr"))
        para.insert(0, ppr)
    return ppr


def style_of(para: ET.Element) -> str:
    ppr = para.find("w:pPr", NS)
    if ppr is None:
        return ""
    p_style = ppr.find("w:pStyle", NS)
    return p_style.get(w("val"), "") if p_style is not None else ""


def set_spacing(ppr: ET.Element, **attrs: str) -> None:
    spacing = ppr.find("w:spacing", NS)
    if spacing is None:
        spacing = ET.Element(w("spacing"))
        ppr.insert(0, spacing)
    for key in list(spacing.attrib):
        del spacing.attrib[key]
    for key, val in attrs.items():
        spacing.set(w(key), val)


def set_ind(ppr: ET.Element, **attrs: str) -> None:
    ind = ppr.find("w:ind", NS)
    if ind is None:
        ind = ET.Element(w("ind"))
        ppr.append(ind)
    for key in list(ind.attrib):
        del ind.attrib[key]
    for key, val in attrs.items():
        ind.set(w(key), val)


def set_jc(ppr: ET.Element, val: str) -> None:
    jc = ppr.find("w:jc", NS)
    if jc is None:
        jc = ET.SubElement(ppr, w("jc"))
    jc.set(w("val"), val)


def has_code_shading(para: ET.Element) -> bool:
    ppr = para.find("w:pPr", NS)
    return ppr is not None and ppr.find("w:shd", NS) is not None


def is_heading_style(style: str) -> bool:
    return style in {"2", "3", "4", "Heading1", "Heading2", "Heading3", "Heading4"}


def main() -> None:
    target = Path(r"C:\Users\hhhh\Desktop\潘敏姿的论文_图文规范版.docx")
    temp = target.with_suffix(".spacing.docx")
    rewritten = target.with_suffix(".spacing.rewritten.docx")
    shutil.copy2(target, temp)

    with zipfile.ZipFile(temp, "r") as zin:
        root = ET.fromstring(zin.read("word/document.xml"))
        body = root.find("w:body", NS)
        if body is None:
            raise RuntimeError("document body not found")
        children = list(body)
        body_start = next(
            i for i, child in enumerate(children)
            if text_of(child) == "1 绪论" and is_heading_style(style_of(child))
        )

        for child in children[body_start:]:
            if child.tag != w("p"):
                continue
            text = text_of(child)
            ppr = ppr_of(child)
            style = style_of(child)
            has_image = child.find(".//w:drawing", NS) is not None

            if has_image:
                set_spacing(ppr, before="120", after="0")
                set_jc(ppr, "center")
                continue

            if text.startswith("图") or text.startswith("代码"):
                set_spacing(ppr, before="110", after="0", line="300", lineRule="auto")
                set_ind(ppr, left="705", right="0", firstLine="0")
                set_jc(ppr, "center")
                continue

            if has_code_shading(child):
                set_spacing(ppr, before="0", after="0", line="240", lineRule="auto")
                set_ind(ppr, left="420")
                continue

            if is_heading_style(style):
                # Keep copied heading spacing from the template.
                continue

            if text:
                set_spacing(ppr, before="168", line="300", lineRule="auto")
                set_ind(ppr, left="808", right="522", firstLine="479")
                set_jc(ppr, "both")

        new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        with zipfile.ZipFile(rewritten, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data = new_xml
                zout.writestr(item, data)

    shutil.move(str(rewritten), str(target))
    temp.unlink(missing_ok=True)
    print(f"normalized={target}")


if __name__ == "__main__":
    main()
