from PyPDF2 import PdfFileReader, PdfFileWriter

# Load the existing PDF and the plot PDF
existing_pdf_path = 'pt-d128.pdf'  # Replace with your actual path
plot_pdf_path = 'transparent_plot.pdf'          # Replace with your actual path
existing_pdf = PdfFileReader(open(existing_pdf_path, 'rb'))
plot_pdf = PdfFileReader(open(plot_pdf_path, 'rb'))

# Assuming you want to overlay the first page of the plot PDF
plot_page = plot_pdf.getPage(0)

# Create a new PDF to save the result
output_pdf = PdfFileWriter()

# Overlay the plot PDF onto the existing PDF
for page_num in range(existing_pdf.getNumPages()):
    # Get the existing page
    existing_page = existing_pdf.getPage(page_num)

    # Adjust these parameters for scaling and positioning
    scale = 0.7145      # Scale factor (e.g., 0.5 is half size)
    tx = 27.5           # Horizontal translation
    ty = 34.5           # Vertical translation

    # Merge the plot page with the existing page
    existing_page.mergeScaledTranslatedPage(plot_page, scale, tx, ty, expand=False)

    # Add the merged page to the output PDF
    output_pdf.addPage(existing_page)

# Save the final output
modified_pdf_path = 'pt-d2-curve.pdf'  # Replace with your desired path
with open(modified_pdf_path, 'wb') as f:
    output_pdf.write(f)
