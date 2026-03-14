"""
CLI entry point

Converts draw.io files to PowerPoint presentations
"""
import sys
import argparse
from pathlib import Path

from .load import DrawIOLoader
from .draw import PPTXWriter
from .data import get_logger, ConversionLogger
# from .analysis import compare_conversion
from .data import ConversionConfig, default_config

# import debugpy
# try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
    # pass
def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='draw.ioファイルをPowerPointプレゼンテーションに変換',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  drawio2pptx input.drawio output.pptx
  drawio2pptx input.drawio output.pptx --analyze
  drawio2pptx input.drawio output.pptx -a
        """
    )
    parser.add_argument('-i','--input', type=str, default="/home/hanjunyi/project/IMG2XML/output/page_010/page_010_merged.drawio.xml",help='入力draw.ioファイルのパス')
    parser.add_argument('-o','--output', type=str, default="output8.pptx", help='出力PowerPointファイルのパス')
    parser.add_argument('-a', '--analyze', action='store_true',
                       help='変換後に解析結果を表示')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Parsing: {input_path}")
    
    try:
        # Create conversion configuration
        config = ConversionConfig()  # Use default settings
        
        # Create logger with config
        logger = ConversionLogger(config=config)
        
        # Load draw.io file
        loader = DrawIOLoader(logger=logger, config=config)
        diagrams = loader.load_file(input_path)
        
        if not diagrams:
            print("No diagrams found in file")
            sys.exit(1)
        
        # Get page size (from first diagram)
        page_size = loader.extract_page_size(diagrams[0])
        
        # Create PowerPoint presentation
        writer = PPTXWriter(logger=logger, config=config)
        prs, blank_layout = writer.create_presentation(page_size)
        
        # Process each diagram
        slide_count = 0
        for mgm in diagrams:
            # Extract elements
            elements = loader.extract_elements(mgm)
            
            # Add to slide
            writer.add_slide(prs, blank_layout, elements)
            slide_count += 1
        
        # Save
        prs.save(output_path)
        print(f"Saved {output_path} ({slide_count} slides)")
        
        # Display warnings
        warnings = logger.get_warnings()
        if warnings:
            print(f"\nWarnings ({len(warnings)}):")
            for warning in warnings:
                print(f"  - {warning.message}")
        
        # Execute analysis if analysis option is specified
        if args.analyze:
            compare_conversion(input_path, output_path)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
