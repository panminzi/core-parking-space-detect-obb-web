from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

ET.register_namespace("w", W_NS)
ET.register_namespace("r", R_NS)

NS = {"w": W_NS}


def q(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def w(name: str) -> str:
    return q(W_NS, name)


def text_of(elem: ET.Element) -> str:
    return "".join(t.text or "" for t in elem.findall(".//w:t", NS)).strip()


def rel_id_max(rels_root: ET.Element) -> int:
    max_id = 0
    for rel in rels_root.findall(q(REL_NS, "Relationship")):
        rid = rel.get("Id", "")
        if rid.startswith("rId") and rid[3:].isdigit():
            max_id = max(max_id, int(rid[3:]))
    return max_id


def add_rel(rels_root: ET.Element, rid: str, rel_type: str, target: str) -> None:
    rel = ET.SubElement(rels_root, q(REL_NS, "Relationship"))
    rel.set("Id", rid)
    rel.set("Type", rel_type)
    rel.set("Target", target)


def ensure_centered_para(para: ET.Element) -> None:
    ppr = para.find("w:pPr", NS)
    if ppr is None:
        ppr = ET.Element(w("pPr"))
        para.insert(0, ppr)
    jc = ppr.find("w:jc", NS)
    if jc is None:
        jc = ET.SubElement(ppr, w("jc"))
    jc.set(w("val"), "center")


def repair_document_xml(document_xml: bytes, header_rid: str, footer_rid: str) -> bytes:
    root = ET.fromstring(document_xml)

    # Restore header/footer references on all section definitions in the generated document.
    for sect in root.findall(".//w:sectPr", NS):
        for child in list(sect):
            if child.tag in {w("headerReference"), w("footerReference")}:
                sect.remove(child)
        header = ET.Element(w("headerReference"))
        header.set(w("type"), "default")
        header.set(q(R_NS, "id"), header_rid)
        footer = ET.Element(w("footerReference"))
        footer.set(w("type"), "default")
        footer.set(q(R_NS, "id"), footer_rid)
        sect.insert(0, footer)
        sect.insert(0, header)

        pg_mar = sect.find("w:pgMar", NS)
        if pg_mar is not None:
            pg_mar.set(w("header"), "899")
            pg_mar.set(w("footer"), "1249")

    # Make inserted visual material centered. Code captions and figure captions are centered,
    # while code lines keep readable left alignment inside an indented shaded block.
    for para in root.findall(".//w:p", NS):
        text = text_of(para)
        has_drawing = para.find(".//w:drawing", NS) is not None
        if has_drawing or text.startswith("图") or text.startswith("代码"):
            ensure_centered_para(para)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main() -> None:
    target = Path(r"C:\Users\hhhh\Desktop\潘敏姿的论文_图文规范版.docx")
    template = Path(r"C:\Users\hhhh\Desktop\2.毕业设计（论文）_QQ浏览器转格式.docx")
    temp = target.with_suffix(".repairing.docx")
    rewritten = target.with_suffix(".rewritten.docx")

    shutil.copy2(target, temp)

    with zipfile.ZipFile(template, "r") as template_zip:
        header1 = template_zip.read("word/header1.xml")
        header2 = template_zip.read("word/header2.xml")
        footer1 = template_zip.read("word/footer1.xml")
        footer2 = template_zip.read("word/footer2.xml")

    with zipfile.ZipFile(temp, "r") as zin:
        rels_root = ET.fromstring(zin.read("word/_rels/document.xml.rels"))
        next_id = rel_id_max(rels_root) + 1
        header_rid = f"rId{next_id}"
        footer_rid = f"rId{next_id + 1}"
        next_id += 2
        add_rel(
            rels_root,
            header_rid,
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header",
            "header2.xml",
        )
        add_rel(
            rels_root,
            footer_rid,
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer",
            "footer2.xml",
        )
        new_rels = ET.tostring(rels_root, encoding="utf-8", xml_declaration=True)
        new_doc = repair_document_xml(zin.read("word/document.xml"), header_rid, footer_rid)

        with zipfile.ZipFile(rewritten, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename in {
                    "word/document.xml",
                    "word/_rels/document.xml.rels",
                    "word/header1.xml",
                    "word/header2.xml",
                    "word/footer1.xml",
                    "word/footer2.xml",
                }:
                    continue
                zout.writestr(item, zin.read(item.filename))
            zout.writestr("word/document.xml", new_doc)
            zout.writestr("word/_rels/document.xml.rels", new_rels)
            zout.writestr("word/header1.xml", header1)
            zout.writestr("word/header2.xml", header2)
            zout.writestr("word/footer1.xml", footer1)
            zout.writestr("word/footer2.xml", footer2)

    shutil.move(str(rewritten), str(target))
    temp.unlink(missing_ok=True)
    print(f"repaired={target}")


if __name__ == "__main__":
    main()
