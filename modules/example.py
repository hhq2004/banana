"""
模块使用示例

重构后的工作流：
    1. SAM3提取 → 输出ElementInfo（bbox, mask, type等）
    2. 各子模块处理 → 每个子模块生成自己的mxCell XML
    3. XMLMerger → 收集、排序、合并所有XML片段
"""

import os
import sys

# 确保可以导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import (
    # 基础类
    ProcessingContext,
    
    # 数据类型
    ElementInfo,
    BoundingBox,
    ProcessingResult,
    XMLFragment,
    LayerLevel,
    get_layer_level,
    
    # 处理模块
    Sam3InfoExtractor,      # 任务1
    IconPictureProcessor,   # 任务2
    BasicShapeProcessor,    # 任务3
    OtherShapeProcessor,    # 任务4
    ArrowProcessor,         # 任务5
    XMLMerger,              # 任务6
    MetricEvaluator,        # 任务7
    RefinementProcessor,    # 任务8
    
    # 文字处理模块（另一个小组）
    TextProcessor,
    create_text_fragments,
    
    # 快捷函数
    merge_fragments,
)


# ======================== 新架构示例 ========================

def example_new_workflow():
    """
    展示重构后的工作流程
    
    核心变化：
        - 每个子模块负责生成自己的XML
        - XMLMerger只负责收集和合并
    """
    print("=" * 60)
    print("重构后的工作流程示例")
    print("=" * 60)
    
    # 模拟各子模块生成的XML片段
    fragments = []
    
    # ===== 任务3: 基本图形模块生成的XML =====
    # 大的矩形（背景容器）- 层级0，会在最底层
    fragments.append(XMLFragment(
        element_id=0,
        xml_content='''<mxCell id="0" parent="1" vertex="1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;">
  <mxGeometry x="50" y="50" width="500" height="400" as="geometry"/>
</mxCell>''',
        layer_level=LayerLevel.BACKGROUND.value,
        bbox=BoundingBox(50, 50, 550, 450),
        element_type="section_panel"
    ))
    
    # 普通矩形 - 层级1
    fragments.append(XMLFragment(
        element_id=1,
        xml_content='''<mxCell id="1" parent="1" vertex="1" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;">
  <mxGeometry x="100" y="100" width="120" height="60" as="geometry"/>
</mxCell>''',
        layer_level=LayerLevel.BASIC_SHAPE.value,
        bbox=BoundingBox(100, 100, 220, 160),
        element_type="rectangle"
    ))
    
    # 椭圆形 - 层级1
    fragments.append(XMLFragment(
        element_id=2,
        xml_content='''<mxCell id="2" parent="1" vertex="1" value="" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;">
  <mxGeometry x="300" y="100" width="120" height="60" as="geometry"/>
</mxCell>''',
        layer_level=LayerLevel.BASIC_SHAPE.value,
        bbox=BoundingBox(300, 100, 420, 160),
        element_type="ellipse"
    ))
    
    # ===== 任务2: Icon模块生成的XML =====
    # 假设这是一个去背景后的icon图片
    fake_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    fragments.append(XMLFragment(
        element_id=3,
        xml_content=f'''<mxCell id="3" parent="1" vertex="1" value="" style="shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;aspect=fixed;image=data:image/png,{fake_base64};">
  <mxGeometry x="150" y="200" width="40" height="40" as="geometry"/>
</mxCell>''',
        layer_level=LayerLevel.IMAGE.value,
        bbox=BoundingBox(150, 200, 190, 240),
        element_type="icon"
    ))
    
    # ===== 任务5: 箭头模块生成的XML =====
    fragments.append(XMLFragment(
        element_id=4,
        xml_content=f'''<mxCell id="4" parent="1" vertex="1" value="" style="shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;aspect=fixed;image=data:image/png,{fake_base64};">
  <mxGeometry x="220" y="115" width="80" height="30" as="geometry"/>
</mxCell>''',
        layer_level=LayerLevel.ARROW.value,
        bbox=BoundingBox(220, 115, 300, 145),
        element_type="arrow"
    ))
    
    # ===== 文字模块生成的XML（如果有OCR） =====
    fragments.append(XMLFragment(
        element_id=5,
        xml_content='''<mxCell id="5" parent="1" vertex="1" value="开始" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;">
  <mxGeometry x="130" y="110" width="60" height="30" as="geometry"/>
</mxCell>''',
        layer_level=LayerLevel.TEXT.value,
        bbox=BoundingBox(130, 110, 190, 140),
        element_type="text"
    ))
    
    fragments.append(XMLFragment(
        element_id=6,
        xml_content='''<mxCell id="6" parent="1" vertex="1" value="结束" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;">
  <mxGeometry x="330" y="110" width="60" height="30" as="geometry"/>
</mxCell>''',
        layer_level=LayerLevel.TEXT.value,
        bbox=BoundingBox(330, 110, 390, 140),
        element_type="text"
    ))
    
    # ===== 任务6: XMLMerger合并 =====
    print("\n[任务6] XMLMerger合并...")
    print(f"  收集到 {len(fragments)} 个XML片段")
    
    # 显示排序前的顺序
    print("\n  排序前:")
    for f in fragments:
        print(f"    - ID={f.element_id}, Type={f.element_type}, Layer={f.layer_level}")
    
    # 使用快捷函数合并
    output_path = merge_fragments(
        fragments=fragments,
        canvas_width=600,
        canvas_height=500,
        output_path="./output/example_merged.drawio.xml"
    )
    
    print(f"\n  排序后（按layer_level升序，同层按面积降序）:")
    sorted_fragments = sorted(fragments, key=lambda f: (f.layer_level, -f.area))
    for f in sorted_fragments:
        print(f"    - ID={f.element_id}, Type={f.element_type}, Layer={f.layer_level}, Area={f.area}")
    
    print(f"\n  XML已保存: {output_path}")
    
    return output_path


def example_submodule_xml_generation():
    """
    展示子模块如何生成XML
    
    每个子模块的职责：
        1. 读取元素信息（bbox, mask等）
        2. 处理元素（取色/去背景/etc）
        3. 生成mxCell XML字符串
        4. 设置 element.xml_fragment 和 element.layer_level
    """
    print("\n" + "=" * 60)
    print("子模块生成XML示例")
    print("=" * 60)
    
    # 模拟SAM3提取的元素
    element = ElementInfo(
        id=0,
        element_type="rectangle",
        bbox=BoundingBox(100, 100, 220, 160),
        score=0.95,
        polygon=[[100,100], [220,100], [220,160], [100,160]]
    )
    
    print(f"\n[SAM3提取] 元素信息:")
    print(f"  - Type: {element.element_type}")
    print(f"  - BBox: ({element.bbox.x1}, {element.bbox.y1}, {element.bbox.x2}, {element.bbox.y2})")
    print(f"  - Score: {element.score}")
    
    # 模拟BasicShapeProcessor处理
    print(f"\n[BasicShapeProcessor] 处理元素:")
    
    # 1. 提取颜色（模拟）
    fill_color = "#dae8fc"
    stroke_color = "#6c8ebf"
    print(f"  - 提取颜色: fill={fill_color}, stroke={stroke_color}")
    
    # 2. 生成XML
    style = f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill_color};strokeColor={stroke_color};"
    xml_fragment = f'''<mxCell id="{element.id}" parent="1" vertex="1" value="" style="{style}">
  <mxGeometry x="{element.bbox.x1}" y="{element.bbox.y1}" width="{element.bbox.width}" height="{element.bbox.height}" as="geometry"/>
</mxCell>'''
    
    print(f"  - 生成XML:")
    print(f"    {xml_fragment[:80]}...")
    
    # 3. 设置元素属性
    element.xml_fragment = xml_fragment
    element.layer_level = get_layer_level(element.element_type)
    element.fill_color = fill_color
    element.stroke_color = stroke_color
    
    print(f"  - 设置layer_level: {element.layer_level} (BASIC_SHAPE)")
    print(f"  - element.has_xml(): {element.has_xml()}")
    
    return element


def example_layer_levels():
    """
    展示层级系统
    """
    print("\n" + "=" * 60)
    print("层级系统说明")
    print("=" * 60)
    
    print("""
DrawIO的Z轴规则：先写入的在底层，后写入的在顶层

层级定义（从底到顶）：
┌─────────────────────────────────────────────────────────┐
│  LayerLevel.OTHER (5)       - 其他（默认最上层）         │
├─────────────────────────────────────────────────────────┤
│  LayerLevel.TEXT (4)        - 文字                      │
├─────────────────────────────────────────────────────────┤
│  LayerLevel.ARROW (3)       - 箭头/连接线               │
├─────────────────────────────────────────────────────────┤
│  LayerLevel.IMAGE (2)       - 图片类（icon, picture）   │
├─────────────────────────────────────────────────────────┤
│  LayerLevel.BASIC_SHAPE (1) - 基本图形                  │
├─────────────────────────────────────────────────────────┤
│  LayerLevel.BACKGROUND (0)  - 背景/容器（最底层）       │
└─────────────────────────────────────────────────────────┘

排序规则：
    1. 首先按 layer_level 升序（小的在底层）
    2. 同层级内按面积降序（大的在下，避免遮挡小元素）
""")
    
    # 展示get_layer_level函数
    test_types = [
        "section_panel", "title_bar",  # BACKGROUND
        "rectangle", "ellipse", "diamond",  # BASIC_SHAPE
        "icon", "picture", "chart",  # IMAGE
        "arrow", "line",  # ARROW
        "text",  # TEXT
        "unknown",  # OTHER
    ]
    
    print("元素类型 → 层级映射:")
    for t in test_types:
        level = get_layer_level(t)
        level_name = LayerLevel(level).name
        print(f"  {t:20s} → {level} ({level_name})")


def example_full_pipeline_new():
    """
    完整的处理流程示例（新架构）
    """
    print("\n" + "=" * 60)
    print("完整处理流程（新架构）")
    print("=" * 60)
    
    print("""
流程图：

┌─────────────────────────────────────────────────────────────┐
│                        输入图片                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  任务1: SAM3提取信息                                         │
│  输出: List[ElementInfo] (bbox, mask, type, score)          │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ 任务2: Icon/Pic │  │ 任务3: 基本图形 │  │ 任务5: 箭头     │
│                 │  │                 │  │                 │
│ 处理:           │  │ 处理:           │  │ 处理:           │
│ - RMBG去背景    │  │ - 提取颜色      │  │ - 方向检测      │
│ - 转base64      │  │ - 生成style     │  │ - 去背景        │
│                 │  │                 │  │                 │
│ 输出:           │  │ 输出:           │  │ 输出:           │
│ - xml_fragment  │  │ - xml_fragment  │  │ - xml_fragment  │
│ - layer=IMAGE   │  │ - layer=BASIC   │  │ - layer=ARROW   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  任务6: XMLMerger                                            │
│                                                              │
│  职责（只做这些）：                                           │
│    1. 收集所有 xml_fragment                                  │
│    2. 按 layer_level 排序                                    │
│    3. 同层级按面积降序                                        │
│    4. 重新分配ID                                             │
│    5. 生成完整DrawIO XML                                     │
│                                                              │
│  不做：                                                      │
│    - 不生成样式                                              │
│    - 不决定颜色                                              │
│    - 不处理base64                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      输出 DrawIO XML                         │
└─────────────────────────────────────────────────────────────┘
""")


def example_text_team_integration():
    """
    展示文字小组如何与系统集成
    """
    print("\n" + "=" * 60)
    print("文字小组集成示例")
    print("=" * 60)
    
    # ===== 模拟文字小组的OCR结果 =====
    print("\n[文字小组] OCR识别结果:")
    
    ocr_results = [
        {'text': '开始', 'bbox': [130, 110, 190, 140], 'font_size': 14},
        {'text': '结束', 'bbox': [330, 110, 390, 140], 'font_size': 14},
        {'text': '处理流程', 'bbox': [200, 50, 320, 80], 'font_size': 16, 'align': 'center'},
        {'text': '$f(x) = \\int_0^1 x^2 dx$', 'bbox': [150, 300, 350, 350], 'is_latex': True},
    ]
    
    for r in ocr_results:
        print(f"  - '{r['text']}' at {r['bbox']}")
    
    # ===== 方式1: 使用便捷函数 =====
    print("\n[方式1] 使用 create_text_fragments():")
    
    text_fragments = create_text_fragments(ocr_results, canvas_width=800, canvas_height=600)
    
    print(f"  生成了 {len(text_fragments)} 个文字XML片段")
    for f in text_fragments:
        print(f"    - layer={f.layer_level} (TEXT), bbox={f.bbox.to_list() if f.bbox else 'N/A'}")
    
    # ===== 方式2: 手动创建XMLFragment =====
    print("\n[方式2] 手动创建XMLFragment:")
    print('''
    from modules import XMLFragment, LayerLevel, BoundingBox
    
    fragment = XMLFragment(
        element_id=0,
        xml_content=\'\'\'<mxCell id="0" parent="1" vertex="1" value="文字" 
                style="text;html=1;...">
          <mxGeometry x="100" y="100" width="60" height="20" as="geometry"/>
        </mxCell>\'\'\',
        layer_level=LayerLevel.TEXT.value,  # ⭐ 关键：设为TEXT (4)
        bbox=BoundingBox(100, 100, 160, 120),
        element_type="text"
    )
    ''')
    
    # ===== 与图形XML合并 =====
    print("\n[合并] 文字XML + 图形XML:")
    
    # 模拟图形小组的XML片段
    shape_fragments = [
        XMLFragment(
            element_id=0,
            xml_content='<mxCell id="0" parent="1" vertex="1" value="" style="rounded=0;..."><mxGeometry x="100" y="100" width="120" height="60" as="geometry"/></mxCell>',
            layer_level=LayerLevel.BASIC_SHAPE.value,
            bbox=BoundingBox(100, 100, 220, 160),
            element_type="rectangle"
        ),
    ]
    
    # 合并所有片段
    all_fragments = shape_fragments + text_fragments
    
    print(f"  图形片段: {len(shape_fragments)} 个 (layer=1)")
    print(f"  文字片段: {len(text_fragments)} 个 (layer=4)")
    print(f"  合计: {len(all_fragments)} 个")
    
    print("\n  排序后顺序（layer升序）:")
    sorted_all = sorted(all_fragments, key=lambda f: f.layer_level)
    for f in sorted_all:
        layer_name = {0: 'BACKGROUND', 1: 'BASIC_SHAPE', 2: 'IMAGE', 3: 'ARROW', 4: 'TEXT', 5: 'OTHER'}.get(f.layer_level, 'UNKNOWN')
        print(f"    - Type={f.element_type}, Layer={f.layer_level} ({layer_name})")
    
    print("\n  → 图形先写入（底层），文字后写入（顶层）")
    print("  → 文字不会被图形遮挡 ✓")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="模块使用示例（新架构）")
    parser.add_argument("--mode", "-m", default="all",
                        choices=["all", "workflow", "submodule", "layers", "text"],
                        help="运行模式")
    
    args = parser.parse_args()
    
    if args.mode == "all":
        example_full_pipeline_new()
        example_layer_levels()
        example_submodule_xml_generation()
        example_new_workflow()
        example_text_team_integration()
    elif args.mode == "workflow":
        example_new_workflow()
    elif args.mode == "submodule":
        example_submodule_xml_generation()
    elif args.mode == "layers":
        example_layer_levels()
    elif args.mode == "text":
        example_text_team_integration()