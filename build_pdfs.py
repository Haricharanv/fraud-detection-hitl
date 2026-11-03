"""
build_pdfs.py -- High-Resolution PDF Builder with KaTeX Math Rendering
Samsung Capstone Project

Converts Markdown documents to publication-grade PDFs.
Extracts math expressions, preserves LaTeX syntax, renders via KaTeX inside Playwright Chromium.
"""

import asyncio
import re
from pathlib import Path
import markdown
from playwright.async_api import async_playwright

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>

<!-- KaTeX Stylesheet & Scripts -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function() {{
    renderMathInElement(document.body, {{
        delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$', right: '$', display: false}}
        ],
        throwOnError: false
    }});
}});
</script>

<style>
    @page {{
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
        @bottom-right {{
            content: counter(page);
        }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1e293b;
        line-height: 1.55;
        font-size: 10pt;
        background: #ffffff;
        margin: 0;
        padding: 0;
    }}
    h1 {{
        font-size: 20pt;
        color: #0f172a;
        border-bottom: 2.5px solid #2563eb;
        padding-bottom: 6px;
        margin-top: 0;
        margin-bottom: 10px;
        font-weight: 700;
        letter-spacing: -0.4px;
    }}
    h2 {{
        font-size: 14pt;
        color: #1e3a8a;
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 8px;
        font-weight: 600;
    }}
    h3 {{
        font-size: 11.5pt;
        color: #1d4ed8;
        margin-top: 14px;
        margin-bottom: 5px;
        font-weight: 600;
    }}
    h4 {{
        font-size: 10.5pt;
        color: #334155;
        margin-top: 12px;
        margin-bottom: 4px;
        font-weight: 600;
    }}
    p {{
        margin-top: 0;
        margin-bottom: 8px;
    }}
    strong {{
        color: #0f172a;
    }}
    blockquote {{
        border-left: 3.5px solid #3b82f6;
        background: #f8fafc;
        padding: 8px 14px;
        margin: 10px 0;
        color: #334155;
        font-style: italic;
        border-radius: 0 4px 4px 0;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 9pt;
        page-break-inside: avoid;
    }}
    th {{
        background: #f1f5f9;
        color: #0f172a;
        font-weight: 600;
        text-align: left;
        padding: 6px 9px;
        border: 1px solid #cbd5e1;
    }}
    td {{
        padding: 6px 9px;
        border: 1px solid #e2e8f0;
        vertical-align: top;
    }}
    tr:nth-child(even) {{
        background: #f8fafc;
    }}
    code {{
        font-family: "Cascadia Code", Consolas, "Courier New", monospace;
        background: #f1f5f9;
        color: #b91c1c;
        padding: 1.5px 4.5px;
        border-radius: 3px;
        font-size: 8.5pt;
    }}
    pre {{
        background: #0f172a;
        color: #f8fafc;
        padding: 12px;
        border-radius: 5px;
        overflow-x: auto;
        font-size: 8pt;
        line-height: 1.4;
        page-break-inside: avoid;
        margin: 12px 0;
    }}
    pre code {{
        background: transparent;
        color: inherit;
        padding: 0;
    }}
    ul, ol {{
        margin-top: 0;
        margin-bottom: 8px;
        padding-left: 20px;
    }}
    li {{
        margin-bottom: 3px;
    }}
    hr {{
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 18px 0;
    }}
    .katex {{
        font-size: 1.05em;
    }}
    .katex-display {{
        margin: 0.8em 0;
        padding: 4px 0;
        overflow-x: auto;
        overflow-y: hidden;
    }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def process_markdown_with_math(md_text: str) -> str:
    """Protect math blocks before markdown conversion, then restore them."""
    placeholders = {}

    def save_display_math(match):
        token = f"ZDISPXYZ{len(placeholders)}XYZ"
        placeholders[token] = f"$${match.group(1)}$$"
        return token

    def save_inline_math(match):
        token = f"ZINLINEXYZ{len(placeholders)}XYZ"
        placeholders[token] = f"${match.group(1)}$"
        return token

    # 1. Protect display math $$ ... $$
    protected = re.sub(r"\$\$(.*?)\$\$", save_display_math, md_text, flags=re.DOTALL)

    # 2. Protect inline math $ ... $ (excluding escaped \$)
    protected = re.sub(r"(?<!\\)\$([^\$\n]+?)(?<!\\)\$", save_inline_math, protected)

    # 3. Convert Markdown to HTML
    html = markdown.markdown(
        protected,
        extensions=["tables", "fenced_code", "sane_lists"]
    )

    # 4. Restore original math expressions
    for token, math_expr in placeholders.items():
        html = html.replace(token, math_expr)

    return html


async def convert_md_to_pdf(browser, md_path: Path, pdf_path: Path, title: str):
    print(f"[INFO] Converting {md_path.name} -> {pdf_path.name} ...")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_content = process_markdown_with_math(md_text)
    full_html = HTML_TEMPLATE.format(title=title, body=html_content)

    temp_html = md_path.with_suffix(".html")
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(full_html)

    page = await browser.new_page()
    await page.set_content(full_html, wait_until="networkidle")

    # Give KaTeX a moment to render all equations
    await page.wait_for_timeout(1200)

    katex_count = await page.locator(".katex").count()
    print(f"       Rendered {katex_count} KaTeX math elements in {pdf_path.name}")

    await page.pdf(
        path=str(pdf_path),
        format="A4",
        print_background=True,
        margin={
            "top": "18mm",
            "bottom": "18mm",
            "left": "16mm",
            "right": "16mm"
        },
        display_header_footer=True,
        header_template='<div style="font-size: 7.5pt; color: #94a3b8; width: 100%; text-align: right; padding-right: 16mm;">Samsung Capstone Project | Enterprise Fraud Detection</div>',
        footer_template='<div style="font-size: 7.5pt; color: #94a3b8; width: 100%; text-align: center;">Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>'
    )
    await page.close()

    if temp_html.exists():
        temp_html.unlink()

    print(f"[SUCCESS] Generated: {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")


async def main():
    print("=" * 60)
    print("  BUILDING HIGH-RESOLUTION PDF ARTIFACTS (KaTeX + Playwright)")
    print("=" * 60)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            PROJECT_ROOT / "PRESENTATION_GUIDE.md",
            OUTPUTS_DIR / "presentation_guide.pdf",
            "Samsung Capstone - Presentation Guide & Defense Script"
        ),
        (
            PROJECT_ROOT / "PROJECT_REPORT.md",
            OUTPUTS_DIR / "capstone_project_report.pdf",
            "Samsung Capstone - Final Technical Report"
        )
    ]

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for md_path, pdf_path, title in tasks:
            await convert_md_to_pdf(browser, md_path, pdf_path, title)
        await browser.close()

    print("\n[COMPLETE] Both PDF documents successfully generated with full KaTeX math formatting.\n")


if __name__ == "__main__":
    asyncio.run(main())
