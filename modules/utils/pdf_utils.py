from typing import Optional, List
import os

def pdf_to_images(pdf_path: str, images_dir: str, dpi: int = 96) -> List[str]:
    """
    将PDF转换为PNG图片
    
    Args:
        pdf_path: PDF文件路径
        images_dir: 图片输出目录
        dpi: 渲染分辨率
        
    Returns:
        图片路径列表
    """
    import fitz
    
    doc = fitz.open(pdf_path)
    image_paths: List[str] = []
    
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        img_path = os.path.join(images_dir, f"page_{page_index + 1:03d}.png")
        pix.save(str(img_path))
        image_paths.append(str(img_path))
    
    doc.close()
    print(f" [pdf_to_images] 渲染了 {len(image_paths)} 页")
    return image_paths