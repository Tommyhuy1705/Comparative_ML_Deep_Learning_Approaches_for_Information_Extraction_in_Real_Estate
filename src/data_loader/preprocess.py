import re

def clean_and_format_content(text):
    if not isinstance(text, str): return ""
    
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'(^|[\n])\s*[\*\-\+•]\s+', r'\1. ', text)
    text = re.sub(r':\s*[\*\-\+•]\s+', ': ', text) # # Xử lý trường hợp dấu bullet nằm sau dấu hai chấm (VD: "Tiện ích: + Hồ bơi")
    text = re.sub(r'(LH|Liên hệ|Gọi ngay|Hotline|Zalo|Phone|liên hệ|lh|gọi ngay|hotline|zalo|mobile|phone)[:\s]*[\d\.\s]+', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'(\d)\.(\d)', r'\1_DOT_\2', text)
    text = re.sub(r'[^\w\s,.\(\)\"\':;%+\-]+', ' ', text)
    
    
    sentences = re.split(r'[.\n]+', text)
    
    cleaned_sentences = []
    for s in sentences:
        s = s.strip()
        s = s.lstrip(' -+*•.,')
        s = re.sub(r'\s+', ' ', s)

        s = s.replace('_DOT_', ',')
        cleaned_sentences.append(s)
            
    return "\n".join(cleaned_sentences)

def clean_title(text):
    if not isinstance(text, str): return "No Title"
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



