import json

TOPIC_KEYWORDS = {
    "Gia đình & Xưng hô": [
        "anh", "em", "chị", "bố", "ba", "mẹ", "má", "ông", "bà", "con", "cháu", "cô", "dì",
        "chú", "bác", "cậu", "mợ", "vợ", "chồng", "gia đình", "họ hàng", "bạn", "tôi", "mình",
        "người", "trẻ em", "thanh niên", "phụ nữ", "đàn ông", "bà ngoại", "bà nội", "ông ngoại", "ông nội"
    ],
    "Thời gian": [
        "ngày", "đêm", "sáng", "trưa", "chiều", "tối", "tuần", "tháng", "năm", "giờ", "phút",
        "giây", "hôm nay", "hôm qua", "ngày mai", "bây giờ", "sớm", "muộn", "ban ngày", "ban đêm",
        "buổi sáng", "buổi trưa", "buổi chiều", "buổi tối", "mùa", "xuân", "hạ", "thu", "đông", "tết"
    ],
    "Chào hỏi & Giao tiếp": [
        "chào", "xin chào", "cảm ơn", "xin lỗi", "tạm biệt", "vâng", "dạ", "không", "có",
        "mời", "chúc", "hỏi", "trả lời", "nói", "kể", "gọi", "tên", "tuổi", "địa chỉ", "gặp", "quen"
    ],
    "Y tế & Sức khỏe": [
        "bệnh", "viện", "bác sĩ", "y tá", "thuốc", "đau", "ốm", "sốt", "ho", "khám", "tiêm",
        "y tế", "sức khỏe", "khỏe", "mệt", "cấp cứu", "băng", "máu", "tim", "phổi", "răng", "mắt", "tai"
    ],
    "Tình yêu & Cảm xúc": [
        "yêu", "thích", "ghét", "vui", "buồn", "giận", "sợ", "lo", "nhớ", "thương", "hạnh phúc",
        "an ủi", "cười", "khóc", "ngạc nhiên", "xấu hổ", "tự hào", "hy vọng", "thất vọng", "cảm xúc", "tình yêu"
    ],
    "Đồ vật & Đời sống": [
        "bàn", "ghế", "tủ", "giường", "nhà", "cửa", "xe", "sách", "vở", "bút", "điện thoại",
        "máy tính", "bàn phím", "tivi", "quần", "áo", "giày", "dép", "mũ", "nón", "ba lô", "cặp",
        "tiền", "cơm", "nước", "trà", "cà phê", "bia", "bánh", "hoa", "cây", "chó", "mèo"
    ],
    "Hành động": [
        "ăn", "uống", "đi", "đứng", "ngồi", "nằm", "chạy", "nhảy", "ngủ", "thức", "học", "đọc",
        "viết", "vẽ", "nghe", "nhìn", "xem", "làm", "chơi", "mua", "bán", "tặng", "biếu", "cho",
        "nhận", "giúp", "mở", "đóng", "cầm", "nắm", "ném", "bắt", "nấu", "rửa", "giặt", "quét"
    ]
}

def classify_word(word):
    w_lower = word.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw == w_lower or f" {kw} " in f" {w_lower} ":
                return topic
    return "Từ vựng mở rộng"

def main():
    with open("models/label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)

    topics_dict = {topic: [] for topic in TOPIC_KEYWORDS.keys()}
    topics_dict["Từ vựng mở rộng"] = []

    for word in sorted(label_map.keys()):
        topic = classify_word(word)
        topics_dict[topic].append(word)

    # Loại bỏ chủ đề rỗng nếu có
    topics_dict = {k: v for k, v in topics_dict.items() if len(v) > 0}

    with open("web/models/topics.json", "w", encoding="utf-8") as f:
        json.dump(topics_dict, f, ensure_ascii=False, indent=2)

    for k, v in topics_dict.items():
        print(f"- {k}: {len(v)} từ")

if __name__ == "__main__":
    main()