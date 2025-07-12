import os
import sys
import re
from pathlib import Path

# Try importing different PDF libraries in order of preference
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

# OCR libraries for scanned PDFs
try:
    import pytesseract
    from pdf2image import convert_from_path
    HAS_OCR = True
    
    # Try to set Tesseract path for Windows
    if os.name == 'nt':  # Windows
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\AppData\Local\Tesseract-OCR\tesseract.exe'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
        else:
            print("Warning: Tesseract not found in common Windows locations")
            print("Please install Tesseract or set the path manually")
    
except ImportError:
    HAS_OCR = False

def check_tesseract_installation():
    """Check if Tesseract is properly installed and accessible"""
    if not HAS_OCR:
        return False, "pytesseract or pdf2image not installed"
    
    try:
        # Try to get Tesseract version
        version = pytesseract.get_tesseract_version()
        
        # Check if Bengali language is available
        langs = pytesseract.get_languages()
        has_bengali = 'ben' in langs
        
        return True, f"Tesseract {version} installed. Bengali support: {'Yes' if has_bengali else 'No'}"
    except Exception as e:
        return False, f"Tesseract error: {str(e)}"

def clean_mixed_text(text):
    """Clean and format mixed Bengali-English text for better readability"""
    if not text:
        return ""
    
    # First, normalize line breaks and whitespace
    text = re.sub(r'\r\n|\r', '\n', text)
    
    # Split into lines for processing
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            processed_lines.append("")  # Keep empty lines for paragraph breaks
            continue
            
        # Fix internal word spacing issues for both languages
        line = fix_mixed_word_spacing(line)
        
        # Handle broken sentences - look for incomplete lines
        if line and not any(line.endswith(ending) for ending in ['।', '?', '!', '।।', '.', ';', ':']):
            # Check if line seems complete based on common patterns
            if len(line) > 20:
                # Bengali endings
                if line.endswith(('র', 'ে', 'ত', 'য়', 'ন', 'স', 'ল', 'ক')):
                    line += '।'
                # English endings  
                elif line.endswith(('ed', 'ing', 'ion', 'ly', 'er', 'est', 'th', 'st', 'nd', 'rd')):
                    line += '.'
        
        processed_lines.append(line)
    
    # Join lines back
    text = '\n'.join(processed_lines)
    
    # Fix common mixed language text issues
    text = fix_mixed_formatting(text)
    
    return text

def fix_mixed_word_spacing(line):
    """Fix spacing issues for mixed Bengali-English text"""
    if not line:
        return line
    
    # Remove excessive spaces first
    line = re.sub(r'\s+', ' ', line)
    
    # Fix broken Bengali words - common patterns
    # 1. Consonant + virama + consonant combinations that got separated
    line = re.sub(r'([ক-হ])্\s+([ক-হ])', r'\1্\2', line)
    
    # 2. Base character + dependent vowel signs that got separated
    line = re.sub(r'([ক-হঅ-ঔ])\s+([া-ৌ])', r'\1\2', line)
    
    # 3. Character + nukta/hasant that got separated
    line = re.sub(r'([ক-হ])\s+([়্])', r'\1\2', line)
    
    # 4. Fix ref/ra-phala combinations
    line = re.sub(r'([ক-হ])্\s+র', r'\1্র', line)
    line = re.sub(r'র্\s+([ক-হ])', r'র্\1', line)
    
    # 5. Fix numbers that got separated (both Bengali and English)
    line = re.sub(r'([০-৯])\s+([০-৯])', r'\1\2', line)
    line = re.sub(r'([0-9])\s+([0-9])', r'\1\2', line)
    
    # 6. Fix currency and units that got separated
    line = re.sub(r'([০-৯0-9])\s+(টাকা|পয়সা|কিলো|লিটার|মিটার|kg|km|cm|mm|Rs|USD|EUR)', r'\1 \2', line)
    
    # 7. Fix common English contractions
    line = re.sub(r"([a-zA-Z])\s+(')\s*([a-zA-Z])", r"\1'\3", line)  # don't, can't, etc.
    
    # 8. Fix broken English words (common OCR errors)
    line = re.sub(r'\b([A-Z])\s+([a-z]+)', r'\1\2', line)  # Broken capitalized words
    
    # 9. Ensure proper spacing between words (single space)
    line = re.sub(r'([ক-হঅ-ঔ০-৯।?!a-zA-Z0-9])\s+([ক-হঅ-ঔ০-৯a-zA-Z])', r'\1 \2', line)
    
    return line.strip()

def fix_mixed_formatting(text):
    """Fix common mixed language formatting issues"""
    
    # Remove extra spaces around punctuation (both Bengali and English)
    text = re.sub(r'\s+([।?!.,;:])', r'\1', text)
    
    # Add proper space after sentence endings
    text = re.sub(r'([।?!.,;:])([ক-হঅ-ঔ০-৯a-zA-Z0-9])', r'\1 \2', text)
    
    # Fix spacing around quotation marks
    text = re.sub(r'([।?!.,;:])\s*["\']\s*([ক-হঅ-ঔa-zA-Z])', r'\1 "\2', text)
    text = re.sub(r'([ক-হঅ-ঔ।?!.,;:a-zA-Z])\s*["\']\s*', r'\1" ', text)
    
    # Fix broken compound words (Bengali conjuncts)
    conjuncts = ['ক্ষ', 'জ্ঞ', 'ঞ্চ', 'ঞ্জ', 'ত্র', 'ন্দ', 'ন্ত', 'ন্থ', 'ন্ধ', 'ম্প', 'ম্ব', 'ম্ম', 'ল্প', 'ল্ল', 'শ্চ', 'শ্ত', 'স্থ', 'স্প', 'স্ত', 'হ্ন', 'হ্ম', 'হ্য', 'হ্র', 'হ্ল', 'হ্ব', 'ন্ত্র']
    
    for conjunct in conjuncts:
        parts = conjunct.split('্')
        if len(parts) == 2:
            # Fix separated conjuncts
            pattern = parts[0] + r'্\s+' + parts[1]
            text = re.sub(pattern, conjunct, text)
    
    # Fix paragraph spacing - ensure double line breaks between paragraphs
    text = re.sub(r'([।?!.,;:])\s*\n\s*([ক-হঅ-ঔa-zA-Z])', r'\1\n\n\2', text)
    
    # Clean up multiple consecutive empty lines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text

def fix_word_spacing(line):
    """Fix spacing issues within words and between words"""
    if not line:
        return line
    
    # Remove excessive spaces first
    line = re.sub(r'\s+', ' ', line)
    
    # Fix broken Bengali words - common patterns
    # 1. Consonant + virama + consonant combinations that got separated
    line = re.sub(r'([ক-হ])্\s+([ক-হ])', r'\1্\2', line)
    
    # 2. Base character + dependent vowel signs that got separated
    line = re.sub(r'([ক-হঅ-ঔ])\s+([া-ৌ])', r'\1\2', line)
    
    # 3. Character + nukta/hasant that got separated
    line = re.sub(r'([ক-হ])\s+([়্])', r'\1\2', line)
    
    # 4. Fix ref/ra-phala combinations
    line = re.sub(r'([ক-হ])্\s+র', r'\1্র', line)
    line = re.sub(r'র্\s+([ক-হ])', r'র্\1', line)
    
    # 5. Fix numbers that got separated
    line = re.sub(r'([০-৯])\s+([০-৯])', r'\1\2', line)
    
    # 6. Fix currency and units that got separated
    line = re.sub(r'([০-৯])\s+(টাকা|পয়সা|কিলো|লিটার|মিটার)', r'\1 \2', line)
    
    # 7. Ensure proper spacing between words (single space)
    line = re.sub(r'([ক-হঅ-ঔ০-৯।?!])\s+([ক-হঅ-ঔ০-৯])', r'\1 \2', line)
    
    return line.strip()

def fix_bengali_formatting(text):
    """Fix common Bengali formatting issues"""
    
    # Remove extra spaces around Bengali punctuation
    text = re.sub(r'\s+([।?!])', r'\1', text)
    
    # Add proper space after sentence endings
    text = re.sub(r'([।?!])([ক-হঅ-ঔ০-৯])', r'\1 \2', text)
    
    # Fix spacing around quotation marks
    text = re.sub(r'([।?!])\s*"\s*([ক-হঅ-ঔ])', r'\1 "\2', text)
    text = re.sub(r'([ক-হঅ-ঔ।?!])\s*"\s*', r'\1" ', text)
    
    # Fix broken compound words (more comprehensive)
    # Handle separated conjuncts
    conjuncts = ['ক্ষ', 'জ্ঞ', 'ঞ্চ', 'ঞ্জ', 'ত্র', 'ন্দ', 'ন্ত', 'ন্থ', 'ন্ধ', 'ম্প', 'ম্ব', 'ম্ম', 'ল্প', 'ল্ল', 'শ্চ', 'শ্ত', 'স্থ', 'স্প', 'স্ত', 'হ্ন', 'হ্ম', 'হ্য', 'হ্র', 'হ্ল', 'হ্ব', 'ন্ত্র']
    
    for conjunct in conjuncts:
        parts = conjunct.split('্')
        if len(parts) == 2:
            # Fix separated conjuncts
            pattern = parts[0] + r'্\s+' + parts[1]
            text = re.sub(pattern, conjunct, text)
    
    # Fix paragraph spacing - ensure double line breaks between paragraphs
    text = re.sub(r'([।?!])\s*\n\s*([ক-হঅ-ঔ])', r'\1\n\n\2', text)
    
    # Clean up multiple consecutive empty lines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text

def reconstruct_bengali_paragraphs(text):
    """Reconstruct proper paragraphs from fragmented text"""
    lines = text.split('\n')
    reconstructed = []
    current_paragraph = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines but track them
        if not line:
            if current_paragraph:
                # Join current paragraph with proper spacing
                para_text = ' '.join(current_paragraph)
                para_text = ensure_proper_word_spacing(para_text)
                reconstructed.append(para_text)
                current_paragraph = []
            
            # Add paragraph break
            if reconstructed and reconstructed[-1] != "":
                reconstructed.append("")
            i += 1
            continue
        
        # Check if line starts a new section/paragraph
        if (line.startswith('অধ্যায়') or 
            line.startswith('পরিচ্ছেদ') or
            line.startswith('ভূমিকা') or
            line.startswith('উপসংহার') or
            line.startswith('সূচনা') or
            re.match(r'^[০-৯]+[\.\)]', line) or  # Numbered sections
            re.match(r'^[ক-ৱ][\.\)]', line) or   # Lettered sections
            re.match(r'^[০-৯]+\s*[।:]', line)):  # Numbered with Bengali punctuation
            
            # Save current paragraph if exists
            if current_paragraph:
                para_text = ' '.join(current_paragraph)
                para_text = ensure_proper_word_spacing(para_text)
                reconstructed.append(para_text)
                current_paragraph = []
            
            # Add section break
            if reconstructed and reconstructed[-1] != "":
                reconstructed.append("")
            
            reconstructed.append(line)
            reconstructed.append("")  # Add space after section header
            i += 1
            continue
        
        # Check if this line continues the previous sentence
        if current_paragraph and not line.endswith(('।', '?', '!', '।।')):
            # Check if previous line also doesn't end with punctuation
            prev_line = current_paragraph[-1] if current_paragraph else ""
            if prev_line and not prev_line.endswith(('।', '?', '!', '।।')):
                # This might be a continuation of the same sentence
                if len(line) < 100:  # Short lines are likely continuations
                    current_paragraph.append(line)
                else:
                    # Long line, might be a new paragraph
                    if current_paragraph:
                        para_text = ' '.join(current_paragraph)
                        para_text = ensure_proper_word_spacing(para_text)
                        reconstructed.append(para_text)
                        current_paragraph = []
                    current_paragraph.append(line)
            else:
                current_paragraph.append(line)
        else:
            current_paragraph.append(line)
        
        # If sentence ends, consider paragraph complete
        if line.endswith(('।', '?', '!', '।।')):
            # Look ahead to see if next line starts a new thought
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and (next_line[0].isupper() or 
                                next_line.startswith(('তাই', 'সুতরাং', 'কিন্তু', 'কিন্তু', 'এবং', 'আর', 'তবে', 'যদিও', 'যেহেতু'))):
                    # Complete current paragraph
                    if current_paragraph:
                        para_text = ' '.join(current_paragraph)
                        para_text = ensure_proper_word_spacing(para_text)
                        reconstructed.append(para_text)
                        current_paragraph = []
                    reconstructed.append("")  # Paragraph break
        
        i += 1
    
    # Add remaining paragraph
    if current_paragraph:
        para_text = ' '.join(current_paragraph)
        para_text = ensure_proper_word_spacing(para_text)
        reconstructed.append(para_text)
    
    # Join and clean up
    result = '\n'.join(reconstructed)
    
    # Clean up excessive empty lines
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
    
    return result

def ensure_proper_word_spacing(text):
    """Ensure proper spacing between words in a paragraph"""
    if not text:
        return text
    
    # Split by spaces and rejoin with single spaces
    words = text.split()
    cleaned_words = []
    
    for word in words:
        word = word.strip()
        if word:
            cleaned_words.append(word)
    
    # Join with single spaces
    result = ' '.join(cleaned_words)
    
    # Final cleanup for Bengali-specific spacing
    result = fix_word_spacing(result)
    
    return result

def extract_with_pymupdf(pdf_path):
    """Extract text using PyMuPDF with better formatting"""
    doc = fitz.open(pdf_path)
    text_content = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Try different extraction methods for better spacing
        # Method 1: Try with dict format for better spacing info
        blocks = page.get_text("dict")
        page_text = ""
        
        if blocks and "blocks" in blocks:
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        if "spans" in line:
                            for span in line["spans"]:
                                if "text" in span:
                                    line_text += span["text"]
                        if line_text.strip():
                            page_text += line_text + "\n"
                    page_text += "\n"  # Add space between blocks
        
        # Fallback to regular text extraction if dict method fails
        if not page_text.strip():
            page_text = page.get_text("text")
            
            # If still no text, try blocks method
            if not page_text.strip():
                blocks = page.get_text("blocks")
                if isinstance(blocks, list):
                    for block in blocks:
                        if len(block) > 4 and block[4].strip():
                            page_text += block[4] + "\n\n"
        
        if page_text and page_text.strip():
            cleaned_text = clean_mixed_text(page_text)
            text_content += f"--- পৃষ্ঠা {page_num + 1} ---\n\n"
            text_content += cleaned_text + "\n\n"
    
    doc.close()
    return text_content

def extract_with_pdfplumber(pdf_path):
    """Extract text using pdfplumber with better formatting"""
    text_content = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = ""
            
            # Try multiple extraction methods
            # Method 1: Extract with layout preservation
            try:
                page_text = page.extract_text(layout=True, x_tolerance=2, y_tolerance=2)
            except:
                page_text = ""
            
            # Method 2: Try word-based extraction for better spacing
            if not page_text or len(page_text.strip()) < 50:
                try:
                    words = page.extract_words()
                    if words:
                        # Group words by lines based on their y-coordinates
                        lines = {}
                        for word in words:
                            y = round(word['top'], 1)
                            if y not in lines:
                                lines[y] = []
                            lines[y].append(word)
                        
                        # Sort lines by y-coordinate and create text
                        sorted_lines = sorted(lines.items())
                        for y, line_words in sorted_lines:
                            line_words.sort(key=lambda w: w['x0'])  # Sort words by x-coordinate
                            line_text = ' '.join([w['text'] for w in line_words])
                            page_text += line_text + "\n"
                except:
                    pass
            
            # Method 3: Fallback to regular extraction
            if not page_text or len(page_text.strip()) < 10:
                page_text = page.extract_text()
            
            if page_text and page_text.strip():
                cleaned_text = clean_mixed_text(page_text)
                text_content += f"--- পৃষ্ঠা {page_num + 1} ---\n\n"
                text_content += cleaned_text + "\n\n"
    
    return text_content

def extract_with_pypdf2(pdf_path):
    """Extract text using PyPDF2 with formatting fixes"""
    text_content = ""
    
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            
            if page_text and page_text.strip():
                cleaned_text = clean_mixed_text(page_text)
                text_content += f"--- পৃষ্ঠা {page_num + 1} ---\n"
                text_content += cleaned_text + "\n\n"
    
    return text_content

def extract_with_ocr(pdf_path, language='ben+eng'):
    """Extract text using OCR with post-processing for mixed Bengali-English text"""
    print("OCR দিয়ে টেক্সট এক্সট্র্যাক্ট করা হচ্ছে... (বাংলা + ইংরেজি)")
    
    # Check Tesseract installation first
    tesseract_ok, message = check_tesseract_installation()
    if not tesseract_ok:
        print(f"Tesseract সমস্যা: {message}")
        return ""
    
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)  # Higher DPI for better OCR
        text_content = ""
        
        for page_num, image in enumerate(images):
            try:
                # Configure Tesseract for better mixed language recognition
                # PSM 6 assumes uniform block of text
                # PSM 3 for fully automatic page segmentation (better for mixed content)
                custom_config = r'--oem 3 --psm 3'
                
                # Use Tesseract OCR with mixed language support
                page_text = pytesseract.image_to_string(image, lang=language, config=custom_config)
                
                if page_text and page_text.strip():
                    cleaned_text = clean_mixed_text(page_text)
                    text_content += f"--- পৃষ্ঠা {page_num + 1} ---\n"
                    text_content += cleaned_text + "\n\n"
            except Exception as e:
                print(f"পৃষ্ঠা {page_num + 1} এর জন্য OCR ব্যর্থ: {e}")
        
        return text_content
        
    except Exception as e:
        print(f"OCR প্রক্রিয়ায় ত্রুটি: {e}")
        return ""

def detect_if_scanned(pdf_path):
    """Check if PDF is likely scanned (image-based)"""
    if not HAS_PYMUPDF:
        return False
    
    try:
        doc = fitz.open(pdf_path)
        text_ratio = 0
        total_pages = min(3, len(doc))
        
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            text = page.get_text().strip()
            images = page.get_images()
            
            # If page has images and very little text, likely scanned
            if len(images) > 0 and len(text) < 100:
                text_ratio += 1
        
        doc.close()
        return text_ratio / total_pages > 0.6
    except:
        return False

def pdf_to_txt(pdf_path, txt_path=None, use_ocr=False, preserve_structure=True):
    """
    Convert a PDF file to a text file with proper Bengali formatting.
    
    Args:
        pdf_path (str): Path to the input PDF file
        txt_path (str, optional): Path for the output text file
        use_ocr (bool): Force OCR usage for scanned PDFs
        preserve_structure (bool): Try to preserve document structure
    
    Returns:
        str: Path to the created text file, or None if conversion failed
    """
    try:
        # Validate input
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF ফাইল '{pdf_path}' পাওয়া যায়নি")
        
        # Generate output filename
        if txt_path is None:
            base_name = Path(pdf_path).stem
            txt_path = f"{base_name}_bangla_formatted.txt"
        
        text_content = ""
        
        # Check if we should use OCR
        is_scanned = detect_if_scanned(pdf_path)
        if use_ocr or is_scanned:
            if HAS_OCR:
                text_content = extract_with_ocr(pdf_path)
            else:
                print("OCR লাইব্রেরি পাওয়া যায়নি। ইনস্টল করুন: pip install pytesseract pdf2image")
                return None
        
        # If no text from OCR or not using OCR, try other methods
        if not text_content.strip():
            if HAS_PYMUPDF:
                print("PyMuPDF দিয়ে টেক্সট এক্সট্র্যাক্ট করা হচ্ছে...")
                text_content = extract_with_pymupdf(pdf_path)
            elif HAS_PDFPLUMBER:   
                print("pdfplumber দিয়ে টেক্সট এক্সট্র্যাক্ট করা হচ্ছে...")
                text_content = extract_with_pdfplumber(pdf_path)
            elif HAS_PYPDF2:
                print("PyPDF2 দিয়ে টেক্সট এক্সট্র্যাক্ট করা হচ্ছে...")
                text_content = extract_with_pypdf2(pdf_path)
            else:
                print("কোনো PDF প্রসেসিং লাইব্রেরি পাওয়া যায়নি!")
                return None
        
        # For non-OCR extraction, also clean mixed text
        if not use_ocr and not is_scanned and text_content.strip():
            text_content = clean_mixed_text(text_content)
        
        # Post-process for better structure
        if preserve_structure and text_content.strip():
            text_content = reconstruct_bengali_paragraphs(text_content)
        
        # Handle empty content
        if not text_content.strip():
            print("কোনো টেক্সট পাওয়া যায়নি। এটি স্ক্যান করা PDF হতে পারে।")
            if not use_ocr and HAS_OCR:
                print("স্ক্যান করা PDF এর জন্য --ocr ফ্ল্যাগ ব্যবহার করুন")
            text_content = "PDF থেকে কোনো টেক্সট পাওয়া যায়নি"
        
        # Write with proper encoding
        with open(txt_path, 'w', encoding='utf-16') as txt_file:
            txt_file.write(text_content)
        
        print(f"সফলভাবে রূপান্তরিত: '{pdf_path}' -> '{txt_path}'")
        return txt_path
        
    except Exception as e:
        print(f"রূপান্তরে ত্রুটি: {str(e)}")
        return None

def main():
    """Main function with enhanced Bengali support"""
    print("বাংলা PDF টু টেক্সট রূপান্তরকারী")
    print("=" * 40)
    
    # Check available libraries
    print("উপলব্ধ লাইব্রেরি:")
    print(f"  PyMuPDF: {'✓' if HAS_PYMUPDF else '✗'}")
    print(f"  pdfplumber: {'✓' if HAS_PDFPLUMBER else '✗'}")
    print(f"  PyPDF2: {'✓' if HAS_PYPDF2 else '✗'}")
    
    # Check OCR status
    if HAS_OCR:
        tesseract_ok, message = check_tesseract_installation()
        print(f"  OCR সাপোর্ট: {'✓' if tesseract_ok else '✗'} - {message}")
    else:
        print("  OCR সাপোর্ট: ✗ - pytesseract/pdf2image not installed")
    
    print()
    
    # Handle command line arguments
    use_ocr = '--ocr' in sys.argv
    if use_ocr:
        sys.argv.remove('--ocr')
    
    preserve_structure = '--preserve' in sys.argv
    if preserve_structure:
        sys.argv.remove('--preserve')
    else:
        preserve_structure = True  # Default to preserving structure
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        pdf_file = input("বাংলা PDF ফাইলের পাথ দিন: ").strip()
        
        # Remove quotes
        if pdf_file.startswith('"') and pdf_file.endswith('"'):
            pdf_file = pdf_file[1:-1]
        elif pdf_file.startswith("'") and pdf_file.endswith("'"):
            pdf_file = pdf_file[1:-1]
    
    # Convert PDF
    result = pdf_to_txt(pdf_file, use_ocr=use_ocr, preserve_structure=preserve_structure)
    
    if result:
        print(f"টেক্সট ফাইল সেভ হয়েছে: {os.path.abspath(result)}")
    else:
        print("রূপান্তর ব্যর্থ।")
        print("\nসমস্যা সমাধানের টিপস:")
        print("1. স্ক্যান করা PDF এর জন্য: python script.py filename.pdf --ocr")
        print("2. ভালো লাইব্রেরি ইনস্টল করুন: pip install PyMuPDF pdfplumber")
        print("3. OCR এর জন্য: pip install pytesseract pdf2image")
        print("4. Tesseract ইনস্টল করুন:")
        print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-ben")
        print("   macOS: brew install tesseract tesseract-lang")

if __name__ == "__main__":
    main()