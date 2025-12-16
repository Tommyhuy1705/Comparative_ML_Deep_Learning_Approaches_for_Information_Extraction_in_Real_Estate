import re

PROVINCES = {
    # Miền Bắc
    'hà_nội', 'hải_phòng', 'hải_dương', 'hưng_yên', 'vĩnh_phúc', 'bắc_ninh', 'hà_nam', 'nam_định', 
    'thái_bình', 'ninh_bình', 'hà_giang', 'cao_bằng', 'lào_cai', 'bắc_kạn', 'lạng_sơn', 'tuyên_quang', 
    'yên_bái', 'thái_nguyên', 'phú_thọ', 'bắc_giang', 'quảng_ninh', 'lai_châu', 'điện_biên', 'sơn_la', 'hòa_bình',
    # Miền Trung
    'thanh_hóa', 'nghệ_an', 'hà_tĩnh', 'quảng_bình', 'quảng_trị', 'thừa_thiên_huế', 'đà_nẵng', 
    'quảng_nam', 'quảng_ngãi', 'bình_định', 'phú_yên', 'khánh_hòa', 'ninh_thuận', 'bình_thuận',
    'kon_tum', 'gia_lai', 'đắk_lắk', 'đắk_nông', 'lâm_đồng',
    # Miền Nam & Miền Tây
    'hồ_chí_minh', 'hcm', 'sài_gòn', 'bình_phước', 'bình_dương', 'đồng_nai', 'tây_ninh', 'bà_rịa', 'vũng_tàu',
    'long_an', 'đồng_tháp', 'tiền_giang', 'an_giang', 'bến_tre', 'vĩnh_long', 'trà_vinh', 'hậu_giang', 
    'kiên_giang', 'sóc_trăng', 'bạc_liêu', 'cà_mau', 'cần_thơ'
}

DISTRICTS_KEYWORDS = {
    'quận', 'huyện', 'thị_xã', 'thành_phố', 'tp', 'phường', 'xã', 'thôn', 'ấp', 'khu_phố', 'tổ'
}

REAL_ESTATE_KEYWORDS = {
    'sổ_hồng', 'sổ_đỏ', 'pháp_lý', 'thổ_cư', 'chính_chủ', 'mặt_tiền', 'nở_hậu', 'lô_góc', 'view_biển',
    'full_nội_thất', 'sang_tên', 'công_chứng', 'đặt_cọc', 'giá_bán', 'cho_thuê', 'thương_lượng', 'bao_sang_tên'
}

def get_regex_features(word):
    word_lower = word.lower()
    
    features = {
        'is_province_name': word_lower in PROVINCES,
        'is_district_name': word_lower in DISTRICTS,
        'is_real_estate_term': word_lower in REAL_ESTATE_KEYWORDS,

        'is_numeric': bool(re.match(r'^\d+([.,]\d+)?$', word)),
        'contains_digit': bool(re.search(r'\d', word)),
        'is_price_unit': word_lower in ['tỷ', 'ty', 'tr', 'triệu', 'nghìn', 'ngàn', 'k', 'đ', 'đồng', 'vnd', 'usd'],
        'is_area_unit': word_lower in ['m2', 'm²', 'm', 'mét', 'ha', 'hecta', 'sào', 'công'],
        
        # Bắt dạng: 5x20, 4*15, 5x20m
        'is_dimension': bool(re.match(r'^\d+[\sxX*]\d+[mM]?$', word)),
        
        'is_street_prefix': word_lower in ['đường', 'phố', 'ngõ', 'hẻm', 'ngách', 'hẽm', 'đl', 'ql', 'dt'],
        'is_loc_prefix': word_lower in ['quận', 'huyện', 'thị', 'xã', 'phường', 'tỉnh', 'tp', 'thành', 'khu'],
        'is_project_prefix': word_lower in ['dự', 'án', 'chung', 'cư', 'kdc', 'kcn', 'vincom', 'plaza'],

        'is_house_type': word_lower in ['nhà', 'đất', 'biệt', 'thự', 'căn', 'hộ', 'shophouse', 'officetel', 'kho', 'xưởng', 'lô'],
        
        'is_legal_keyword': word_lower in ['sổ', 'hồng', 'đỏ', 'giấy', 'tờ', 'pháp', 'lý', 'shcc', 'shr'],
        'is_direction': word_lower in ['đông', 'tây', 'nam', 'bắc', 'đb', 'đn', 'tn', 'tb'],
        'is_room_keyword': word_lower in ['phòng', 'pn', 'wc', 'toilet', 'bếp', 'khách', 'ngủ'],
        'is_structure_keyword': word_lower in ['lầu', 'tầng', 'trệt', 'lửng', 'mái', 'hầm'],
        
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
        'word[:3]': word[:3],
    }

    features.update(get_regex_features(word))
    # Context Window +/- 2
    # Lấy features của từ trước và sau
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:is_numeric': bool(re.match(r'^\d+([.,]\d+)?$', word1)),
            '-1:is_loc_prefix': word1.lower() in ['tại', 'ở', 'gần', 'khu', 'đường', 'phố', 'ngõ', 'hẻm', 'quận', 'phường'],
            '-1:is_project_prefix': word1.lower() in ['dự_án', 'chung_cư', 'kdc'],
            '-1:is_house_type': word1.lower() in ['bán', 'mua', 'thuê', 'nhà', 'đất'],
        })
    else:
        features['BOS'] = True

    if i > 1:
        word2 = sent[i-2][0]
        features.update({
            '-2:word.lower()': word2.lower(),
            # Nếu từ -2 là "bán" (bán nhà X) -> X dễ là LOC
            '-2:is_house_type': word2.lower() in ['bán', 'cho_thuê'], 
             # Nếu từ -2 là "tại" (tại đường X) -> X là LOC
            '-2:is_loc_indicator': word2.lower() in ['tại', 'ở', 'gần'],
        })

    # --- Từ phía sau (Next word +1) ---
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:is_title': word1.istitle(),
            '+1:is_numeric': bool(re.match(r'^\d+([.,]\d+)?$', word1)),
            # QUAN TRỌNG: Nhìn sau thấy đơn vị -> chốt ngay số hiện tại
            '+1:is_price_unit': any(x == word1.lower() for x in ['tỷ', 'tr', 'triệu', 'k']),
            '+1:is_area_unit': any(x in word1.lower() for x in ['m2', 'm²', 'm', 'ha']),
            # Nhìn sau thấy tên Quận/Huyện (VD: đường X quận Y) -> X dễ là LOC
            '+1:is_district': word1.lower() in DISTRICTS,
        })
    else:
        features['EOS'] = True # End of Sentence

    # --- Từ phía sau nữa (Next 2nd word +2) ---
    if i < len(sent)-2:
        word2 = sent[i+2][0]
        features.update({
            '+2:word.lower()': word2.lower(),
            # VD: 5 tỷ "đồng" -> Nhìn thấy đồng ở +2 thì số 5 chắc chắn là tiền
            '+2:is_price_unit': word2.lower() in ['đồng', 'vnd'],
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