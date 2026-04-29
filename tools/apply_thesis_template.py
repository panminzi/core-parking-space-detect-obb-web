from __future__ import annotations

import copy
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W_NS)

NS = {"w": W_NS}


def wtag(name: str) -> str:
    return f"{{{W_NS}}}{name}"


def paragraph_text(elem: ET.Element) -> str:
    return "".join(t.text or "" for t in elem.findall(".//w:t", NS)).strip()


def body_paragraph_texts(docx_path: Path) -> list[str]:
    with zipfile.ZipFile(docx_path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    texts: list[str] = []
    for para in root.findall(".//w:p", NS):
        text = paragraph_text(para)
        if text:
            texts.append(text)
    return texts


def set_text_in_element(elem: ET.Element, text: str) -> None:
    texts = elem.findall(".//w:t", NS)
    if not texts:
        return
    texts[0].text = text
    for t in texts[1:]:
        t.text = ""


def clone_ppr_without_numbering(
    template_para: ET.Element,
    *,
    page_break_before: bool = False,
    style_override: str | None = None,
) -> ET.Element | None:
    p_pr = template_para.find("w:pPr", NS)
    if p_pr is None:
        return None
    p_pr = copy.deepcopy(p_pr)
    for num_pr in p_pr.findall("w:numPr", NS):
        p_pr.remove(num_pr)
    if style_override is not None:
        p_style = p_pr.find("w:pStyle", NS)
        if p_style is None:
            p_style = ET.Element(wtag("pStyle"))
            p_pr.insert(0, p_style)
        p_style.set(wtag("val"), style_override)
    if page_break_before and p_pr.find("w:pageBreakBefore", NS) is None:
        p_pr.append(ET.Element(wtag("pageBreakBefore")))
    return p_pr


def make_paragraph(
    template_para: ET.Element,
    text: str,
    *,
    page_break_before: bool = False,
    style_override: str | None = None,
) -> ET.Element:
    para = copy.deepcopy(template_para)
    old_p_pr = para.find("w:pPr", NS)
    for child in list(para):
        para.remove(child)

    p_pr = clone_ppr_without_numbering(
        template_para,
        page_break_before=page_break_before,
        style_override=style_override,
    )
    if p_pr is not None:
        para.append(p_pr)

    run = ET.SubElement(para, wtag("r"))
    old_run = template_para.find("w:r", NS)
    if old_run is not None:
        old_r_pr = old_run.find("w:rPr", NS)
        if old_r_pr is not None:
            run.append(copy.deepcopy(old_r_pr))
    t = ET.SubElement(run, wtag("t"))
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return para


def apply_content_updates(texts: list[str]) -> list[str]:
    updated: list[str] = []
    for text in texts:
        text = text.replace(
            "实验结果表明，本文训练得到的YOLO11-OBB停车位检测模型在验证集上取得Precision为0.96367、Recall为0.96840、mAP@0.5为0.99315、mAP@0.5:0.95为0.97733的结果；在独立测试集上，系统也能够保持较稳定的识别效果。",
            "实验结果表明，本文训练得到的YOLO11-OBB停车位检测模型在验证集上取得Precision约为0.968、Recall约为0.971、mAP@0.5约为0.993、mAP@0.5:0.95约为0.981的结果；在独立测试集上，Precision为0.8885、Recall为0.8228、mAP@0.5为0.8945，说明模型在真实测试样本上具备较好的应用能力。",
        )
        text = text.replace(
            "实验结果表明，本文训练得到的YOLO11-OBB停车位检测模型在验证集上取得Precision为0.96367、Recall为0.96840、mAP@0.5为0.99315、mAP@0.5:0.95为0.97733的结果；在独立测试集上，系统整体Precision为86.90%、Recall为80.30%、mAP@0.5为88.40%、mAP@0.5:0.95为81.30%。",
            "实验结果表明，本文训练得到的YOLO11-OBB停车位检测模型在验证集上取得Precision约为0.968、Recall约为0.971、mAP@0.5约为0.993、mAP@0.5:0.95约为0.981的结果；在独立测试集上，系统整体Precision为88.85%、Recall为82.28%、mAP@0.5为89.45%、mAP@0.5:0.95为78.01%。",
        )
        text = text.replace(
            "在验证集上，训练完成的YOLO11-OBB模型取得Precision为0.96367、Recall为0.96840、mAP@0.5为0.99315、mAP@0.5:0.95为0.97733的结果。这说明模型在验证集上能够较准确地定位停车位并判断状态。较高的mAP@0.5:0.95表明OBB框定位质量较好，模型不仅能够找到目标，也能够较好地拟合目标方向和边界。",
            "在验证集上，训练完成的YOLO11-OBB模型取得Precision约为0.968、Recall约为0.971、mAP@0.5约为0.993、mAP@0.5:0.95约为0.981的结果；在独立测试集上，Precision为88.85%、Recall为82.28%、mAP@0.5为89.45%、mAP@0.5:0.95为78.01%。这说明模型在训练分布附近具备较强检测能力，在独立测试集上仍能保持较好的停车位定位与状态判断效果。",
        )
        text = text.replace(
            "The trained model achieves 0.96367 Precision, 0.96840 Recall, 0.99315 mAP@0.5 and 0.97733 mAP@0.5:0.95 on the validation",
            "The trained model achieves about 0.968 Precision, 0.971 Recall, 0.993 mAP@0.5 and 0.981 mAP@0.5:0.95 on the validation",
        )
        text = re.sub(
            r"本文使用的数据集划分为训练集、验证集和测试集。训练集包含.*?目标实例，其中occupied为.*?个，vacant为.*?个；验证集包含.*?目标实例，其中occupied为.*?个，vacant为.*?个。",
            "本文使用的数据集按照场景分组方式划分为训练集、验证集和测试集，比例约为75%、10%和15%。训练集包含10263张图像、545970个目标实例，其中occupied为261881个，vacant为284089个；验证集包含1368张图像、76182个目标实例，其中occupied为37325个，vacant为38857个；测试集包含2053张图像、113262个目标实例，其中occupied为52904个，vacant为60358个。",
            text,
        )
        text = text.replace("训练轮数设置为50，输入图像尺寸设置为640", "训练轮数设置为80，输入图像尺寸设置为768")
        text = text.replace(
            "第50轮验证集Precision为0.96367，Recall为0.96840，mAP@0.5为0.99315，mAP@0.5:0.95为0.97733",
            "第80轮附近验证集Precision约为0.968，Recall约为0.971，mAP@0.5约为0.993，mAP@0.5:0.95约为0.981",
        )
        updated.append(text)
    return updated


def choose_template(text: str, samples: dict[str, ET.Element], *, in_toc: bool = False) -> ET.Element:
    if in_toc:
        return samples["toc"]
    if text in {"摘  要", "摘要", "Abstract", "目  录", "目 录", "参考文献", "致谢"}:
        return samples["major_title"]
    if re.match(r"^\d+\s+", text):
        return samples["chapter"]
    if re.match(r"^\d+\.\d+\.\d+\s+", text):
        return samples["heading3"]
    if re.match(r"^\d+\.\d+\s+", text):
        return samples["heading2"]
    if text.startswith("关键词") or text.startswith("Key words"):
        return samples["keyword"]
    if re.match(r"^\[\d+\]", text):
        return samples["reference"]
    if re.match(r"^\d+(\.\d+)*\s", text):
        return samples["toc"]
    if re.match(r"^[A-Za-z]", text):
        return samples["english"]
    return samples["normal"]


def split_sections(texts: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    try:
        toc_title_idx = next(i for i, text in enumerate(texts) if text.replace(" ", "") == "目录")
    except StopIteration:
        return texts, [], [], []
    body_start_candidates = [
        i for i in range(toc_title_idx + 1, len(texts))
        if re.match(r"^1\s+绪论$", texts[i])
    ]
    try:
        body_start_idx = body_start_candidates[1] if len(body_start_candidates) > 1 else body_start_candidates[0]
    except IndexError:
        body_start_idx = toc_title_idx + 1
    return texts[:toc_title_idx], [texts[toc_title_idx]], texts[toc_title_idx + 1:body_start_idx], texts[body_start_idx:]


def make_generated_toc(body_texts: list[str]) -> list[str]:
    toc: list[str] = []
    for text in body_texts:
        if re.match(r"^\d+\s+", text) or re.match(r"^\d+\.\d+\s+", text):
            toc.append(text)
        if text == "参考文献":
            toc.append(text)
        if text == "致谢":
            toc.append(text)
    return toc


def is_main_section(text: str) -> bool:
    return bool(re.match(r"^\d+\s+", text)) or text in {"参考文献", "致谢"}


def heading_style_override(text: str, *, in_toc: bool = False) -> str | None:
    if in_toc:
        return None
    if re.match(r"^\d+\s+", text):
        return "Heading1"
    if re.match(r"^\d+\.\d+\.\d+\s+", text):
        return "Heading3"
    if re.match(r"^\d+\.\d+\s+", text):
        return "Heading2"
    return None


def main() -> None:
    template_path = Path(r"C:\Users\hhhh\Desktop\2.毕业设计（论文）_QQ浏览器转格式.docx")
    content_path = Path(r"C:\Users\hhhh\Desktop\潘敏姿的论文.docx")
    output_path = Path(r"C:\Users\hhhh\Desktop\潘敏姿的论文_格式修正版.docx")
    backup_path = Path(r"C:\Users\hhhh\Desktop\潘敏姿的论文_套模板前备份.docx")

    generated_texts = apply_content_updates(body_paragraph_texts(content_path))
    content_from_abstract = generated_texts[7:]
    pre_toc, toc_title, old_toc, body_texts = split_sections(content_from_abstract)
    toc_lines = make_generated_toc(body_texts) if body_texts else old_toc

    if not backup_path.exists():
        shutil.copy2(content_path, backup_path)

    temp_output = output_path.with_name("潘敏姿的论文_套模板处理中.docx")
    shutil.copy2(template_path, temp_output)

    with zipfile.ZipFile(template_path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))

    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("template has no document body")

    elements = list(body)
    sect_pr = elements[-1] if elements and elements[-1].tag == wtag("sectPr") else None
    cover_elements = [copy.deepcopy(e) for e in elements[:25]]

    # Update cover tables while preserving template layout.
    replacements = {
        "基于Yolo v11 的交通标志识别系统": "基于YOLO11-OBB的智慧停车位检测系统设计与实现",
        "202103401431": "",
        "刘新武": "潘敏姿",
        "陈英": "",
        "二○二五年五月": "二〇二六年四月",
    }
    for elem in cover_elements:
        for old, new in replacements.items():
            text = paragraph_text(elem)
            if old in text:
                set_text_in_element(elem, text.replace(old, new))

    samples = {
        "major_title": copy.deepcopy(elements[25]),
        "normal": copy.deepcopy(elements[50]),
        "english": copy.deepcopy(elements[35]),
        "keyword": copy.deepcopy(elements[31]),
        "chapter": copy.deepcopy(elements[47]),
        "heading2": copy.deepcopy(elements[48]),
        "heading3": copy.deepcopy(elements[49]),
        "toc": copy.deepcopy(elements[50]),
        "reference": copy.deepcopy(elements[50]),
    }

    for child in list(body):
        body.remove(child)
    for elem in cover_elements:
        body.append(elem)

    for text in pre_toc:
        body.append(make_paragraph(choose_template(text, samples), text))

    for text in toc_title:
        body.append(make_paragraph(choose_template(text, samples), text, page_break_before=True))

    for text in toc_lines:
        body.append(make_paragraph(choose_template(text, samples, in_toc=True), text))

    for i, text in enumerate(body_texts):
        body.append(
            make_paragraph(
                choose_template(text, samples),
                text,
                page_break_before=is_main_section(text),
                style_override=heading_style_override(text),
            )
        )

    if sect_pr is not None:
        body.append(copy.deepcopy(sect_pr))

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    rewrite_path = temp_output.with_suffix(".rewritten.docx")
    with zipfile.ZipFile(temp_output, "r") as zin, zipfile.ZipFile(rewrite_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = xml_bytes
            zout.writestr(item, data)

    shutil.move(str(rewrite_path), str(output_path))
    if temp_output.exists():
        temp_output.unlink()

    print(f"written={output_path}")
    print(f"backup={backup_path}")
    print(f"paragraphs={len(content_from_abstract)}")


if __name__ == "__main__":
    main()
