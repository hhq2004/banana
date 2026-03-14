"""
CLI entry point

Converts draw.io files to PowerPoint presentations
"""
import sys
import argparse
from pathlib import Path
from typing import Optional

from .load import DrawIOLoader
from .draw import PPTXWriter
from .data import get_logger, ConversionLogger
from .data import ConversionConfig, default_config


class DrawioToPptxConverter:
    """Draw.io to PowerPoint converter"""
    
    def __init__(self, config: Optional[ConversionConfig] = None, logger: Optional[ConversionLogger] = None):
        """
        Initialize converter
        
        Args:
            config: Conversion configuration (uses default_config if None)
            logger: Logger instance (creates new one if None)
        """
        self.config = config or default_config
        self.logger = logger or ConversionLogger(config=self.config)
        self.loader = DrawIOLoader(logger=self.logger, config=self.config)
        self.writer = PPTXWriter(logger=self.logger, config=self.config)
    
    def convert(self, input_path: Path, output_path: Path) -> bool:
        """
        Convert single draw.io file to PowerPoint presentation
        
        Args:
            input_path: Input draw.io file path
            output_path: Output PowerPoint file path
        
        Returns:
            True if conversion succeeded, False otherwise
        """
        if not input_path.exists():
            print(f"Input file not found: {input_path}")
            return False
        
        print(f"Parsing: {input_path}")
        
        try:
            diagrams = self.loader.load_file(input_path)
            
            if not diagrams:
                print("No diagrams found in file")
                return False
            
            page_size = self.loader.extract_page_size(diagrams[0])
            prs, blank_layout = self.writer.create_presentation(page_size)
            
            slide_count = 0
            for mgm in diagrams:
                elements = self.loader.extract_elements(mgm)
                self.writer.add_slide(prs, blank_layout, elements)
                slide_count += 1
            
            prs.save(output_path)
            print(f"Saved {output_path} ({slide_count} slides)")
            
            warnings = self.logger.get_warnings()
            if warnings:
                print(f"\nWarnings ({len(warnings)}):")
                for warning in warnings:
                    print(f"  - {warning.message}")
            
            return True
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def convert_folder(self, folder_path: Path, output_path: Path, pattern: str = "*_merged.drawio.xml") -> bool:
        """
        Convert multiple draw.io files from a folder to a single PowerPoint presentation
        
        Args:
            folder_path: Input folder containing draw.io XML files
            output_path: Output PowerPoint file path
            pattern: File pattern to match (default: "*.xml")
        
        Returns:
            True if conversion succeeded, False otherwise
        """
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"Folder not found or not a directory: {folder_path}")
            return False
        
        xml_files = sorted(folder_path.rglob(pattern))
        
        if not xml_files:
            print(f"No XML files found in folder (including subdirectories): {folder_path}")
            return False
        
        print(f"Found {len(xml_files)} XML files in {folder_path}")
        
        try:
            first_diagrams = self.loader.load_file(xml_files[0])
            if not first_diagrams:
                print(f"No diagrams found in first file: {xml_files[0]}")
                return False
            
            page_size = self.loader.extract_page_size(first_diagrams[0])
            prs, blank_layout = self.writer.create_presentation(page_size)
            
            total_slides = 0
            for xml_file in xml_files:
                print(f"Processing: {xml_file.name}")
                
                try:
                    diagrams = self.loader.load_file(xml_file)
                    
                    if not diagrams:
                        print(f"  Warning: No diagrams found in {xml_file.name}, skipping")
                        continue
                    
                    file_slides = 0
                    for mgm in diagrams:
                        elements = self.loader.extract_elements(mgm)
                        self.writer.add_slide(prs, blank_layout, elements)
                        file_slides += 1
                        total_slides += 1
                    
                    print(f"  Added {file_slides} slide(s)")
                    
                except Exception as e:
                    print(f"  Error processing {xml_file.name}: {e}")
                    continue
            
            prs.save(output_path)
            print(f"\nSaved {output_path} ({total_slides} slides from {len(xml_files)} files)")
            
            warnings = self.logger.get_warnings()
            if warnings:
                print(f"\nWarnings ({len(warnings)}):")
                for warning in warnings:
                    print(f"  - {warning.message}")
            
            return True
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _normalize_output_path(self, input_path: Path, output_path: Path) -> Path:
        """
        Normalize output path based on input path structure
        
        If input is like: output/1/xml/ or output/1/xml
        Then output becomes: output/1/ppt/result.pptx
        
        Args:
            input_path: Input folder path
            output_path: Original output path from args
        
        Returns:
            Normalized output path
        """
        # 如果输出路径不是默认值，直接使用用户指定的路径
        if str(output_path) != "output_merged.pptx":
            return output_path
        
        # 检查输入路径是否符合 output/{name}/xml/ 的结构
        parts = input_path.parts
        
        # 查找 'xml' 目录在路径中的位置
        if 'xml' in parts:
            xml_index = parts.index('xml')
            
            # 如果 xml 是最后一个部分或倒数第二个部分（考虑尾部斜杠）
            if xml_index >= len(parts) - 1:
                # 重构路径：将 xml 替换为 ppt
                parent_parts = parts[:xml_index]
                ppt_dir = Path(*parent_parts) / 'ppt'
                
                # 创建 ppt 目录（如果不存在）
                ppt_dir.mkdir(parents=True, exist_ok=True)
                
                # 返回规范化的输出路径
                return ppt_dir / 'result.pptx'
        
        # 如果不符合预期结构，使用默认行为
        # 输出到输入目录的父目录
        return input_path.parent / 'result.pptx'
    
    def convert_from_xml(self, args) -> bool:
        """
        Convert using parsed arguments (supports both file and folder)
        
        Args:
            args: Parsed command-line arguments
        
        Returns:
            True if conversion succeeded, False otherwise
        """
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        if input_path.is_dir():
            print(f"Input is a folder, processing all XML files...")
            
            # 规范化输出路径
            output_path = self._normalize_output_path(input_path, output_path)
            print(f"Output will be saved to: {output_path}")
            
            pattern = getattr(args, 'pattern', '*.xml')
            return self.convert_folder(input_path, output_path, pattern)
        else:
            return self.convert(input_path, output_path)


def main():
    """Main function (CLI entry point)"""
    parser = argparse.ArgumentParser(
        description='Convert draw.io files to PowerPoint presentations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single file
  python xml2pptx.py -i input.xml -o output.pptx
  
  # Convert a folder (merge all XML files into one PPTX)
  python xml2pptx.py -i /path/to/folder -o output.pptx
  
  # Specify file pattern
  python xml2pptx.py -i /path/to/folder -o output.pptx --pattern "page_*.xml"
        """
    )
    parser.add_argument('-i', '--input', type=str, 
                       default="/home/hanjunyi/project/IMG2XML/output/test_6",
                       help='Input file or folder path')
    parser.add_argument('-o', '--output', type=str, 
                       default="output_merged.pptx", 
                       help='Output PowerPoint file path (auto-detected if input is output/{name}/xml/)')
    parser.add_argument('-p','--pattern', type=str,
                       default="*_merged.drawio.xml",
                       help='File pattern for folder input (default: *.xml)')
    
    args = parser.parse_args()
    
    converter = DrawioToPptxConverter()
    success = converter.convert_from_xml(args)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()