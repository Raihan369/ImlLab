# picture_to_csv.py

from PIL import Image
import pytesseract
import pandas as pd

# -------- Make sure pytesseract and tesseract-ocr are installed via terminal --------
# In your terminal, run:
# pip install pytesseract
# sudo apt update
# sudo apt install -y tesseract-ocr

# Optional: Set tesseract path manually if needed
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Default for many Linux systems

# Load the image
image_path = "new.jpg"  # Adjust path if needed
image = Image.open(image_path)

# Perform OCR to extract text
text = pytesseract.image_to_string(image)

# Optionally, print extracted text
print("Extracted Text:\n", text)

# Process text into rows
rows = text.strip().split("\n")

# Split rows into columns (adjust splitting logic based on your image layout)
data = [row.split() for row in rows if row.strip()]

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("output.csv", index=False)

print("CSV file has been saved as output.csv")
