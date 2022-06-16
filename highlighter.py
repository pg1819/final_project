import fitz
import os
import textwrap

from fpdf import FPDF


def txt_to_pdf(text, filename):
    a4_width_mm = 210
    pt_to_mm = 0.35
    fontsize_pt = 11
    fontsize_mm = fontsize_pt * pt_to_mm
    margin_bottom_mm = 10
    character_width_mm = 7 * pt_to_mm
    width_text = a4_width_mm / character_width_mm

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(True, margin=margin_bottom_mm)
    pdf.add_page()
    pdf.set_font(family='Courier', size=fontsize_pt)
    splitted = text.split('\n')

    for line in splitted:
        lines = textwrap.wrap(line, width_text)
        if len(lines) == 0:
            pdf.ln()
        for wrap in lines:
            pdf.cell(0, fontsize_mm, wrap, ln=1)
    pdf.output(filename, 'F')


def open_pdf(input_pdf):
    return fitz.open(input_pdf)


def highlight_pdf(document, phrase):
    for page in document:
        text_instances = page.search_for(phrase)
        for instance in text_instances:
            highlight = page.add_highlight_annot(instance)
            highlight.update()
    return document


def save_pdf(document, output_pdf):
    document.save(output_pdf, garbage=4, deflate=True, clean=True)


if __name__ == "__main__":
    txt_file = "sample.txt"
    txt_path = os.path.join("./static/uploaded_file", txt_file)

    pdf_file = os.path.splitext(txt_file)[0] + ".pdf"
    pdf_path = os.path.join("./static/results", pdf_file)

    highlighted_pdf_file = "highlighted_" + pdf_file
    highlighted_pdf_path = os.path.join("./static/results", highlighted_pdf_file)

    # with open(txt_path, mode="r", encoding="utf-8-sig") as f:
    #     text = f.read()
    #     txt_to_pdf(text, pdf_path)

    p1 = "quiet, had seemed to ignore him; then she gave in all of a sudden suddenly and so abruptly that, that day, leaving her, radiant and"
    p2 = "--She's playing with it."
    p3 = "This little girl is admirably gifted. But we have to pity her:"
    phrases = [p1, p2, p3]

    doc = open_pdf(pdf_path)
    for ph in phrases:
        doc = highlight_pdf(doc, ph)
    save_pdf(doc, highlighted_pdf_path)
