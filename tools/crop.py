from PyPDF2 import PdfFileReader, PdfFileWriter

def crop_pdf_at_85_percent_width(input_pdf_path, output_pdf_path):
    # Open the existing PDF
    pdf_reader = PdfFileReader(open(input_pdf_path, 'rb'))
    pdf_writer = PdfFileWriter()

    # Process each page
    for page_num in range(pdf_reader.getNumPages()):
        page = pdf_reader.getPage(page_num)

        # Get the page dimensions and convert them to float
        page_width = float(page.mediaBox.upperRight[0])
        page_height = float(page.mediaBox.upperRight[1])

        # Calculate 85% of the width
        new_width = page_width * 0.875

        # Set the new media box (crop the page)
        page.mediaBox.upperRight = (new_width, page_height)

        # Add the cropped page to the new PDF
        pdf_writer.addPage(page)

    # Save the cropped PDF
    with open(output_pdf_path, 'wb') as out_file:
        pdf_writer.write(out_file)

# Paths to your PDFs
input_pdf_path = 'pt-d2-modif.pdf'  # Replace with your input PDF path
output_pdf_path = 'pt-d2-crop.pdf'      # Replace with your desired output PDF path

# Crop the PDF
crop_pdf_at_85_percent_width(input_pdf_path, output_pdf_path)
