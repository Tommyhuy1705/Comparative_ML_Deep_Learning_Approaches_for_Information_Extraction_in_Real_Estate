import re

def get_regex_features(word):
    """
    Hàm trích xuất đặc trưng dựa trên Regex và từ khóa chuyên ngành BĐS.
    """
    word_lower = word.lower()
    
    features = {
        # --- NHÓM SỐ & ĐƠN VỊ (QUAN TRỌNG CHO PRICE/AREA) ---
        'is_numeric': bool(re.match(r'^\d+([.,]\d+)?$', word)), # Bắt số nguyên, thập phân (5.5, 5,5)
        'contains_digit': bool(re.search(r'\d', word)),
        'is_price_unit': word_lower in ['tỷ', 'ty', 'tr', 'triệu', 'nghìn', 'ngàn', 'k', 'đ', 'đồng', 'vnd', 'usd'],
        'is_area_unit': word_lower in ['m2', 'm²', 'm', 'mét', 'ha', 'hecta', 'sào', 'công'],
        
        # --- NHÓM KÍCH THƯỚC (QUAN TRỌNG CHO AREA) ---
        # Bắt dạng: 5x20, 4*15, 5x20m
        'is_dimension': bool(re.match(r'^\d+[\sxX*]\d+[mM]?$', word)),
        
        # --- NHÓM ĐỊA ĐIỂM (QUAN TRỌNG CHO LOC) ---
        'is_street_prefix': word_lower in ['đường', 'phố', 'ngõ', 'hẻm', 'ngách', 'hẽm', 'đl', 'ql', 'dt'],
        'is_loc_prefix': word_lower in ['quận', 'huyện', 'thị', 'xã', 'phường', 'tỉnh', 'tp', 'thành', 'khu'],
        'is_project_prefix': word_lower in ['dự', 'án', 'chung', 'cư', 'kdc', 'kcn', 'vincom', 'plaza'],
        
        # --- NHÓM LOẠI BĐS (QUAN TRỌNG CHO TYPE) ---
        'is_house_type': word_lower in ['nhà', 'đất', 'biệt', 'thự', 'căn', 'hộ', 'shophouse', 'officetel', 'kho', 'xưởng', 'lô'],
        
        # --- NHÓM THUỘC TÍNH (QUAN TRỌNG CHO ATTR) ---
        'is_legal_keyword': word_lower in ['sổ', 'hồng', 'đỏ', 'giấy', 'tờ', 'pháp', 'lý', 'shcc', 'shr'],
        'is_direction': word_lower in ['đông', 'tây', 'nam', 'bắc', 'đb', 'đn', 'tn', 'tb'],
        'is_room_keyword': word_lower in ['phòng', 'pn', 'wc', 'toilet', 'bếp', 'khách', 'ngủ'],
        'is_structure_keyword': word_lower in ['lầu', 'tầng', 'trệt', 'lửng', 'mái', 'hầm'],
        
        # --- ĐẶC TRƯNG HÌNH THÁI TỪ ---
        'is_phone': bool(re.match(r'^(0\d{9}|\d{3,4}\.\d{3}\.\d{3})$', word)),
        'is_all_caps': word.isupper(),
        'is_title': word.istitle(),
        'is_punctuation': word in ['.', ',', '-', ':', '/', '(', ')', '"', '+'],

        'is_compound_word': '_' in word,
        'is_loc_prefix': word_lower in ['quận', 'huyện', 'thị_xã', 'thành_phố', 'tp', 'phường', 'xã'],
        'is_project_prefix': word_lower in ['dự_án', 'chung_cư', 'kdc', 'khu_đô_thị'],
        'is_house_type': word_lower in ['nhà_phố', 'biệt_thự', 'căn_hộ', 'đất_nền', 'nhà_riêng'],
    }
    return features

def word2features(sent, i):
    word = sent[i][0]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        # 'word[-2:]': word[-2:],
        'word[:3]': word[:3], # Tiền tố cũng quan trọng
    }
    
    # 2. Đặc trưng Regex nâng cao (Gọi hàm trên)
    features.update(get_regex_features(word))
    # Context Window +/- 2
    # Lấy features của từ trước và sau
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:is_title': word1.istitle(),
            '-1:is_numeric': bool(re.match(r'^\d+([.,]\d+)?$', word1)),
            '-1:is_street_prefix': word1.lower() in ['đường', 'phố', 'ngõ', 'hẻm'],
            '-1:is_price_unit': word1.lower() in ['tỷ', 'tr', 'triệu'],
            '-1:is_house_type': word1.lower() in ['bán', 'thuê', 'mua', 'nhà', 'đất'],
        })
    else:
        features['BOS'] = True # Đầu câu

    # --- Từ phía trước nữa (Previous 2nd word) ---
    if i > 1:
        word2 = sent[i-2][0]
        features.update({
            '-2:word.lower()': word2.lower(),
            '-2:is_loc_prefix': word2.lower() in ['tại', 'ở', 'gần', 'khu'],
        })

    # --- Từ phía sau (Next word) ---
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:is_title': word1.istitle(),
            '+1:is_numeric': bool(re.match(r'^\d+([.,]\d+)?$', word1)),
            '+1:is_price_unit': word1.lower() in ['tỷ', 'tr', 'triệu', 'k'],
            '+1:is_area_unit': word1.lower() in ['m2', 'm²', 'm'],
        })
    else:
        features['EOS'] = True # Cuối câu

    # --- Từ phía sau nữa (Next 2nd word) ---
    if i < len(sent)-2:
        word2 = sent[i+2][0]
        features.update({
            '+2:word.lower()': word2.lower(),
        })

    return features

def sent2features(sent):
    """Tạo đặc trưng cho cả câu (List of tuples)"""
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    """Lấy nhãn từ list of tuples [(word, label), ...]"""
    return [token[1] for token in sent]

def sent2tokens(sent):
    """Lấy từ từ list of tuples [(word, label), ...]"""
    return [token[0] for token in sent]