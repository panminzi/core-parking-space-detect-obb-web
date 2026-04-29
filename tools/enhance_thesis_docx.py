from __future__ import annotations

import copy
import os
import posixpath
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from PIL import Image


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

for prefix, uri in {
    "w": W_NS,
    "r": R_NS,
    "wp": WP_NS,
    "a": A_NS,
    "pic": PIC_NS,
}.items():
    ET.register_namespace(prefix, uri)

NS = {"w": W_NS, "r": R_NS}


def q(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def w(name: str) -> str:
    return q(W_NS, name)


def text_of(elem: ET.Element) -> str:
    return "".join(t.text or "" for t in elem.findall(".//w:t", NS)).strip()


def first_body_para(root: ET.Element) -> ET.Element:
    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("document body not found")
    for child in body:
        if child.tag == w("p"):
            return child
    raise RuntimeError("no paragraph found")


def clone_para(base: ET.Element, text: str, *, style: str | None = None, align: str | None = None) -> ET.Element:
    para = copy.deepcopy(base)
    for child in list(para):
        para.remove(child)
    ppr = ET.SubElement(para, w("pPr"))
    if style:
        st = ET.SubElement(ppr, w("pStyle"))
        st.set(w("val"), style)
    if align:
        jc = ET.SubElement(ppr, w("jc"))
        jc.set(w("val"), align)
    r = ET.SubElement(para, w("r"))
    t = ET.SubElement(r, w("t"))
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return para


def make_caption(base: ET.Element, text: str) -> ET.Element:
    para = clone_para(base, text, align="center")
    rpr = para.find(".//w:rPr", NS)
    if rpr is None:
        run = para.find("w:r", NS)
        rpr = ET.Element(w("rPr"))
        run.insert(0, rpr)
    sz = ET.SubElement(rpr, w("sz"))
    sz.set(w("val"), "21")
    return para


def make_body_para(base: ET.Element, text: str) -> ET.Element:
    para = clone_para(base, text)
    ppr = para.find("w:pPr", NS)
    spacing = ET.SubElement(ppr, w("spacing"))
    spacing.set(w("line"), "360")
    spacing.set(w("lineRule"), "auto")
    ind = ET.SubElement(ppr, w("ind"))
    ind.set(w("firstLineChars"), "200")
    jc = ET.SubElement(ppr, w("jc"))
    jc.set(w("val"), "both")
    return para


def make_code_para(line: str) -> ET.Element:
    para = ET.Element(w("p"))
    ppr = ET.SubElement(para, w("pPr"))
    spacing = ET.SubElement(ppr, w("spacing"))
    spacing.set(w("before"), "0")
    spacing.set(w("after"), "0")
    spacing.set(w("line"), "240")
    spacing.set(w("lineRule"), "auto")
    ind = ET.SubElement(ppr, w("ind"))
    ind.set(w("left"), "420")
    shd = ET.SubElement(ppr, w("shd"))
    shd.set(w("val"), "clear")
    shd.set(w("color"), "auto")
    shd.set(w("fill"), "F2F2F2")

    run = ET.SubElement(para, w("r"))
    rpr = ET.SubElement(run, w("rPr"))
    fonts = ET.SubElement(rpr, w("rFonts"))
    fonts.set(w("ascii"), "Consolas")
    fonts.set(w("hAnsi"), "Consolas")
    fonts.set(w("eastAsia"), "Consolas")
    sz = ET.SubElement(rpr, w("sz"))
    sz.set(w("val"), "18")
    t = ET.SubElement(run, w("t"))
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = line if line else " "
    return para


def make_image_para(rel_id: str, image_path: Path, doc_pr_id: int, width_inches: float = 5.5) -> ET.Element:
    with Image.open(image_path) as img:
        px_w, px_h = img.size
    emu_per_inch = 914400
    cx = int(width_inches * emu_per_inch)
    cy = int(cx * px_h / px_w)

    para = ET.Element(w("p"))
    ppr = ET.SubElement(para, w("pPr"))
    jc = ET.SubElement(ppr, w("jc"))
    jc.set(w("val"), "center")
    run = ET.SubElement(para, w("r"))
    drawing = ET.SubElement(run, w("drawing"))
    inline = ET.SubElement(drawing, q(WP_NS, "inline"))
    extent = ET.SubElement(inline, q(WP_NS, "extent"))
    extent.set("cx", str(cx))
    extent.set("cy", str(cy))
    effect_extent = ET.SubElement(inline, q(WP_NS, "effectExtent"))
    for key in ["l", "t", "r", "b"]:
        effect_extent.set(key, "0")
    doc_pr = ET.SubElement(inline, q(WP_NS, "docPr"))
    doc_pr.set("id", str(doc_pr_id))
    doc_pr.set("name", f"Picture {doc_pr_id}")
    c_nv = ET.SubElement(inline, q(WP_NS, "cNvGraphicFramePr"))
    locks = ET.SubElement(c_nv, q(A_NS, "graphicFrameLocks"))
    locks.set("noChangeAspect", "1")
    graphic = ET.SubElement(inline, q(A_NS, "graphic"))
    graphic_data = ET.SubElement(graphic, q(A_NS, "graphicData"))
    graphic_data.set("uri", PIC_NS)
    pic = ET.SubElement(graphic_data, q(PIC_NS, "pic"))
    nv_pic_pr = ET.SubElement(pic, q(PIC_NS, "nvPicPr"))
    c_nv_pr = ET.SubElement(nv_pic_pr, q(PIC_NS, "cNvPr"))
    c_nv_pr.set("id", "0")
    c_nv_pr.set("name", image_path.name)
    ET.SubElement(nv_pic_pr, q(PIC_NS, "cNvPicPr"))
    blip_fill = ET.SubElement(pic, q(PIC_NS, "blipFill"))
    blip = ET.SubElement(blip_fill, q(A_NS, "blip"))
    blip.set(q(R_NS, "embed"), rel_id)
    stretch = ET.SubElement(blip_fill, q(A_NS, "stretch"))
    ET.SubElement(stretch, q(A_NS, "fillRect"))
    sp_pr = ET.SubElement(pic, q(PIC_NS, "spPr"))
    xfrm = ET.SubElement(sp_pr, q(A_NS, "xfrm"))
    off = ET.SubElement(xfrm, q(A_NS, "off"))
    off.set("x", "0")
    off.set("y", "0")
    ext = ET.SubElement(xfrm, q(A_NS, "ext"))
    ext.set("cx", str(cx))
    ext.set("cy", str(cy))
    prst = ET.SubElement(sp_pr, q(A_NS, "prstGeom"))
    prst.set("prst", "rect")
    ET.SubElement(prst, q(A_NS, "avLst"))
    return para


def max_rel_id(rels_root: ET.Element) -> int:
    max_id = 0
    for rel in rels_root.findall(q(REL_NS, "Relationship")):
        rid = rel.get("Id", "")
        if rid.startswith("rId") and rid[3:].isdigit():
            max_id = max(max_id, int(rid[3:]))
    return max_id


def add_relationship(rels_root: ET.Element, rel_id: str, target: str) -> None:
    rel = ET.SubElement(rels_root, q(REL_NS, "Relationship"))
    rel.set("Id", rel_id)
    rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
    rel.set("Target", target)


def ensure_content_type(ct_root: ET.Element, ext: str, content_type: str) -> None:
    for default in ct_root.findall(q(CT_NS, "Default")):
        if default.get("Extension") == ext:
            return
    default = ET.SubElement(ct_root, q(CT_NS, "Default"))
    default.set("Extension", ext)
    default.set("ContentType", content_type)


def paragraph_style(elem: ET.Element) -> str:
    ppr = elem.find("w:pPr", NS)
    if ppr is None:
        return ""
    p_style = ppr.find("w:pStyle", NS)
    return p_style.get(w("val"), "") if p_style is not None else ""


def is_real_heading(elem: ET.Element) -> bool:
    style = paragraph_style(elem)
    return style in {"2", "3", "4", "Heading1", "Heading2", "Heading3", "Heading4"}


def make_toc_para(base: ET.Element, title: str, page: str, *, level: int = 1) -> ET.Element:
    para = clone_para(base, "", style="6")
    for child in list(para):
        para.remove(child)
    ppr = ET.SubElement(para, w("pPr"))
    st = ET.SubElement(ppr, w("pStyle"))
    st.set(w("val"), "6")
    tabs = ET.SubElement(ppr, w("tabs"))
    tab = ET.SubElement(tabs, w("tab"))
    tab.set(w("val"), "right")
    tab.set(w("leader"), "dot")
    tab.set(w("pos"), "9000")
    ind = ET.SubElement(ppr, w("ind"))
    ind.set(w("left"), str(0 if level == 1 else 420))
    spacing = ET.SubElement(ppr, w("spacing"))
    spacing.set(w("line"), "360")
    spacing.set(w("lineRule"), "auto")

    run = ET.SubElement(para, w("r"))
    t = ET.SubElement(run, w("t"))
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = title
    run_tab = ET.SubElement(para, w("r"))
    ET.SubElement(run_tab, w("tab"))
    run_page = ET.SubElement(para, w("r"))
    t_page = ET.SubElement(run_page, w("t"))
    t_page.text = page
    return para


def rebuild_toc(body: ET.Element, base: ET.Element) -> None:
    children = list(body)
    toc_idx = next((i for i, child in enumerate(children) if text_of(child) in {"目  录", "目 录"}), None)
    if toc_idx is None:
        return
    body_start_idx = next(
        (
            i for i, child in enumerate(children[toc_idx + 1:], toc_idx + 1)
            if text_of(child) == "1 绪论" and is_real_heading(child)
        ),
        None,
    )
    if body_start_idx is None:
        return

    for _ in range(body_start_idx - toc_idx - 1):
        body.remove(list(body)[toc_idx + 1])

    toc_items = [
        ("1 绪论", "1", 1),
        ("1.1 研究背景与意义", "1", 2),
        ("1.2 国内外研究现状", "2", 2),
        ("1.3 研究内容与技术路线", "3", 2),
        ("1.4 论文组织结构", "3", 2),
        ("2 智慧停车位检测相关技术", "4", 1),
        ("2.1 目标检测基本原理", "4", 2),
        ("2.2 YOLO11与YOLO11-OBB模型", "5", 2),
        ("2.3 定向边界框与停车位检测", "6", 2),
        ("2.4 Flask Web系统开发技术", "7", 2),
        ("2.5 模型评价指标", "7", 2),
        ("3 系统需求分析与总体设计", "9", 1),
        ("3.1 可行性分析", "9", 2),
        ("3.2 功能需求分析", "10", 2),
        ("3.3 非功能需求分析", "10", 2),
        ("3.4 系统总体架构", "11", 2),
        ("3.5 数据流程与模块划分", "12", 2),
        ("4 模型训练与检测算法实现", "14", 1),
        ("4.1 数据集构建与标注规范", "14", 2),
        ("4.2 数据集划分与类别统计", "15", 2),
        ("4.3 YOLO11-OBB模型改进", "16", 2),
        ("4.4 训练配置与过程分析", "17", 2),
        ("4.5 检测后处理与时序稳定策略", "18", 2),
        ("5 智慧停车位检测系统实现", "20", 1),
        ("5.1 后端接口实现", "20", 2),
        ("5.2 图片检测模块实现", "22", 2),
        ("5.3 视频检测模块实现", "25", 2),
        ("5.4 实时检测模块实现", "27", 2),
        ("5.5 模型数据可视化实现", "28", 2),
        ("6 系统测试与结果分析", "31", 1),
        ("6.1 测试环境", "31", 2),
        ("6.2 功能测试", "32", 2),
        ("6.3 模型评价结果", "33", 2),
        ("6.4 误差原因分析", "35", 2),
        ("7 总结与展望", "37", 1),
        ("参考文献", "39", 1),
        ("致谢", "41", 1),
    ]
    insert_at = toc_idx + 1
    for title, page, level in reversed(toc_items):
        body.insert(insert_at, make_toc_para(base, title, page, level=level))


def insert_after_heading(body: ET.Element, heading: str, new_elems: list[ET.Element]) -> None:
    children = list(body)
    for idx, child in enumerate(children):
        if text_of(child) == heading and is_real_heading(child):
            insert_at = idx + 1
            for elem in reversed(new_elems):
                body.insert(insert_at, elem)
            return
    raise RuntimeError(f"heading not found: {heading}")


def code_block(caption: str, code: str, base: ET.Element) -> list[ET.Element]:
    elems = [make_caption(base, caption)]
    elems.extend(make_code_para(line) for line in code.strip("\n").splitlines())
    return elems


def main() -> None:
    repo = Path(r"E:\project\code\core-parking-space-detect-obb-web")
    input_path = Path(r"C:\Users\hhhh\Desktop\潘敏姿的论文_格式修正版.docx")
    output_path = Path(r"C:\Users\hhhh\Desktop\潘敏姿的论文_图文规范版.docx")
    assets = {
        "dashboard": repo / "output/thesis-assets/01_dashboard.png",
        "detect_page": repo / "output/thesis-assets/02_image_detect.png",
        "video_page": repo / "output/thesis-assets/03_video_detect.png",
        "model_data": repo / "output/thesis-assets/04_model_data.png",
        "sample_pred": repo / "other/model_train/detect_obb/output/val/val_batch0_pred.jpg",
        "pr_curve": repo / "other/model_train/detect_obb/output/val/BoxPR_curve.png",
        "confusion": repo / "other/model_train/detect_obb/output/val/confusion_matrix.png",
    }
    missing = [str(path) for path in assets.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))

    tmp = output_path.with_suffix(".tmp.docx")
    shutil.copy2(input_path, tmp)

    with zipfile.ZipFile(input_path, "r") as zin:
        document_xml = zin.read("word/document.xml")
        rels_xml = zin.read("word/_rels/document.xml.rels")
        ct_xml = zin.read("[Content_Types].xml")

    doc_root = ET.fromstring(document_xml)
    rels_root = ET.fromstring(rels_xml)
    ct_root = ET.fromstring(ct_xml)
    body = doc_root.find("w:body", NS)
    if body is None:
        raise RuntimeError("document body not found")
    base = first_body_para(doc_root)
    rebuild_toc(body, base)

    next_rid = max_rel_id(rels_root) + 1
    media_entries: list[tuple[str, Path, str]] = []

    def register_image(key: str) -> str:
        nonlocal next_rid
        path = assets[key]
        ext = path.suffix.lower().lstrip(".")
        rel_id = f"rId{next_rid}"
        next_rid += 1
        target = f"media/thesis_{key}.{ext}"
        add_relationship(rels_root, rel_id, target)
        media_entries.append((target, path, ext))
        if ext in {"jpg", "jpeg"}:
            ensure_content_type(ct_root, ext, "image/jpeg")
        elif ext == "png":
            ensure_content_type(ct_root, ext, "image/png")
        return rel_id

    image_rel = {key: register_image(key) for key in assets}
    doc_pr_id = 100

    def img(key: str, caption: str, width: float = 5.6) -> list[ET.Element]:
        nonlocal doc_pr_id
        doc_pr_id += 1
        return [
            make_image_para(image_rel[key], assets[key], doc_pr_id, width),
            make_caption(base, caption),
        ]

    insert_after_heading(body, "3.4 系统总体架构", [
        make_body_para(base, "系统工作台用于集中展示图片检测、视频检测、实时检测和模型数据查看等入口，用户登录后可从该页面进入各功能模块。系统运行界面如图3.1所示。"),
        *img("dashboard", "图3.1 系统工作台运行界面", 5.7),
    ])

    insert_after_heading(body, "5.1 后端接口实现", [
        make_body_para(base, "后端采用Flask组织页面路由和检测接口，前端将模型名称与图像数据提交到接口后，由服务层完成模型加载、图像解码和OBB检测。核心接口代码如代码5.1所示。"),
        *code_block("代码5.1 Flask图片检测接口核心代码", """
@app.route('/api/detect', methods=['POST'])
def api_detect():
    data = request.json
    model_name = data.get('model', 'ready-model')
    image_data = data.get('image')
    if not image_data:
        return jsonify({'code': 400, 'message': '请提供图像数据'})
    result = detect_objects(model_name, image_data)
    return jsonify(result)
""", base),
    ])

    insert_after_heading(body, "5.2 图片检测模块实现", [
        make_body_para(base, "图片检测页面提供模型选择、图片上传、结果预览和车位状态统计等功能。用户完成上传后点击检测按钮，系统返回带有OBB多边形框的检测结果图。页面运行效果如图5.1所示，检测样例如图5.2所示。"),
        *img("detect_page", "图5.1 图片检测模块运行界面", 5.7),
        *img("sample_pred", "图5.2 停车位检测样例结果图", 5.7),
        *code_block("代码5.2 图像检测服务核心代码", """
def detect_objects(model_name, image_data):
    model = load_model(model_name)
    image = decode_base64_image(image_data)
    optimized_result = run_robust_obb_detection(
        model, temp_path,
        image_size=image.size,
        class_name_mapping=CLASS_NAME_MAPPING,
        strict=True
    )
    detections = optimized_result['detections']
    detection_image = draw_obb_detection_boxes(image, detections)
    return {'code': 200, 'data': {
        'detections': detections,
        'total_detections': len(detections),
        'detection_image': detection_image
    }}
""", base),
    ])

    insert_after_heading(body, "5.3 视频检测模块实现", [
        make_body_para(base, "视频检测模块在逐帧处理的基础上加入固定车位编号与状态防抖策略，能够减少连续帧中偶发漏检或类别跳变带来的视觉闪烁。视频检测页面如图5.3所示。"),
        *img("video_page", "图5.3 视频检测模块运行界面", 5.7),
        *code_block("代码5.3 视频车位状态稳定策略核心代码", """
class ParkingVideoStateStabilizer:
    def update(self, detections, frame_number):
        for detection in detections:
            parking_id = detection.get('parking_space_id')
            raw_state = detection.get('class_name')
            state = self._states.setdefault(parking_id, {
                'stable_state': raw_state,
                'candidate_state': None,
                'candidate_votes': 0
            })
            if raw_state != state['stable_state']:
                state['candidate_votes'] += 1
                if state['candidate_votes'] >= VIDEO_STATE_CHANGE_VOTES:
                    state['stable_state'] = raw_state
            detection['stable_state'] = state['stable_state']
        return detections
""", base),
    ])

    insert_after_heading(body, "5.5 模型数据可视化实现", [
        make_body_para(base, "模型数据页面读取训练和测试输出文件，将Precision、Recall、mAP等指标以卡片、曲线和类别表格形式展示，便于对模型迭代效果进行对比。页面效果如图5.4所示。"),
        *img("model_data", "图5.4 模型数据可视化页面", 5.7),
    ])

    insert_after_heading(body, "6.3 模型评价结果", [
        make_body_para(base, "除指标数值外，本文还结合PR曲线和混淆矩阵观察模型在不同类别上的表现。PR曲线能够反映不同置信度阈值下准确率和召回率的变化关系，混淆矩阵则用于分析occupied与vacant两类之间的误判情况。"),
        *img("pr_curve", "图6.1 模型PR曲线", 5.2),
        *img("confusion", "图6.2 模型混淆矩阵", 5.0),
    ])

    new_document_xml = ET.tostring(doc_root, encoding="utf-8", xml_declaration=True)
    new_rels_xml = ET.tostring(rels_root, encoding="utf-8", xml_declaration=True)
    new_ct_xml = ET.tostring(ct_root, encoding="utf-8", xml_declaration=True)

    rewritten = output_path.with_suffix(".rewritten.docx")
    with zipfile.ZipFile(input_path, "r") as zin, zipfile.ZipFile(rewritten, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = new_document_xml
            elif item.filename == "word/_rels/document.xml.rels":
                data = new_rels_xml
            elif item.filename == "[Content_Types].xml":
                data = new_ct_xml
            zout.writestr(item, data)

        existing = set(zin.namelist())
        for target, src, _ext in media_entries:
            arcname = posixpath.join("word", target)
            if arcname not in existing:
                zout.write(src, arcname)

    shutil.move(str(rewritten), str(output_path))
    if tmp.exists():
        tmp.unlink()

    print(f"written={output_path}")
    print(f"images={len(media_entries)}")


if __name__ == "__main__":
    main()
