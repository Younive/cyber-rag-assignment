import re
def clean_thai_gov_ocr(text):
    # 1. Remove OCR Source Tags (e.g., )
    text = re.sub(r'\[.*?\]', '', text)

    # 2. Remove Page Headers and Footers
    # Removing "Page XX" (หน้า ๓๓)
    text = re.sub(r'หน้า\s+[๐-๙\d]+', '', text)
    # Removing Gazette Header (เล่ม ... ราชกิจจานุเบกษา ...)
    text = re.sub(r'เล่ม\s+[๐-๙\d]+.*?ราชกิจจานุเบกษา.*?[\r\n]+', '', text, flags=re.DOTALL)
    # Removing centered page numbers like "- ๕ -" or "-ไ๒-"
    text = re.sub(r'-\s*[\w๐-๙]+\s*-', '', text)

    # "we." appears to be a misread of "พ.ศ."
    text = text.replace('we.', 'พ.ศ.')
    # "๒๕๒๐๒" is a common OCR error for "๒๕๖๒" (Cybersecurity Act year)
    text = text.replace('๒๕๒๐๒', '๒๕๖๒') 
    # "๒๕๒๐๕" seems to be a misread of "๒๕๖๕"
    text = text.replace('๒๕๒๐๕', '๒๕๖๕')
    
    thai_digits = '๐１２３４５６７８９' # Using wide chars just in case, or standard
    thai_digits_std = '๐๑๒๓๔๕๖๗๘๙'
    arabic_digits = '0123456789'
    trans_table = str.maketrans(thai_digits_std, arabic_digits)
    text = text.translate(trans_table)

    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()