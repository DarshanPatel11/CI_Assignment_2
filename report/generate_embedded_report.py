#!/usr/bin/env python3
"""
Generate REPORT.md with embedded base64 images for self-contained PDF conversion.
"""
import base64
from pathlib import Path

def get_base64_image(image_path: str) -> str:
    """Read image and return base64 encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    report_dir = Path(__file__).parent
    screenshots_dir = report_dir / "screenshots"

    # Read the current report
    report_path = report_dir / "REPORT.md"
    content = report_path.read_text()

    # Image mapping
    images = {
        "screenshots/01_main_interface.jpg": screenshots_dir / "01_main_interface.jpg",
        "screenshots/02_search_results.jpg": screenshots_dir / "02_search_results.jpg",
        "screenshots/03_sources_timing.jpg": screenshots_dir / "03_sources_timing.jpg",
        "screenshots/04_hybrid_results.jpg": screenshots_dir / "04_hybrid_results.jpg",
        "screenshots/05_dense_results.jpg": screenshots_dir / "05_dense_results.jpg",
        "screenshots/06_context_used.jpg": screenshots_dir / "06_context_used.jpg",
        "screenshots/architecture_hybrid_rag.png": screenshots_dir / "architecture_hybrid_rag.png",
        "screenshots/architecture_data_pipeline.png": screenshots_dir / "architecture_data_pipeline.png",
        "screenshots/architecture_evaluation_pipeline.png": screenshots_dir / "architecture_evaluation_pipeline.png",
    }


    # Replace each image path with base64 data URI
    for rel_path, abs_path in images.items():
        if abs_path.exists():
            b64_data = get_base64_image(abs_path)
            # Detect MIME type from extension
            mime_type = "image/png" if str(abs_path).endswith(".png") else "image/jpeg"
            data_uri = f"data:{mime_type};base64,{b64_data}"
            content = content.replace(f"]({rel_path})", f"]({data_uri})")
            print(f"✅ Embedded: {rel_path}")
        else:
            print(f"❌ Missing: {abs_path}")

    # Write the new report
    output_path = report_dir / "REPORT_EMBEDDED.md"
    output_path.write_text(content)
    print(f"\n📄 Created: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
