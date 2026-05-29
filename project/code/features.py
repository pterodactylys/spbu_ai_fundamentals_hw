from __future__ import annotations

import re

import pandas as pd

BRAND_RULES: list[tuple[str, str]] = [
    ("redmi", r"\bredmi\b"),
    ("poco", r"\bpoco\b"),
    ("honor", r"\b(honor|хонор)\b"),
    ("realme", r"\brealme\b"),
    ("oneplus", r"\b(oneplus|one\+\+|one\+)\b"),
    ("apple", r"\b(apple|iphone|айфон)\b"),
    ("google", r"\b(pixel|google pixel)\b"),
    ("samsung", r"\b(samsung|самсунг|galaxy)\b"),
    ("xiaomi", r"\b(xiaomi|ксиаоми|mi\b)"),
    ("huawei", r"\b(huawei|хуавей)\b"),
    ("oppo", r"\boppo\b"),
    ("vivo", r"\bvivo\b"),
    ("sony", r"\bsony\b"),
    ("nokia", r"\bnokia\b"),
    ("motorola", r"\bmotorola\b"),
    ("asus", r"\b(asus|zenfone|rog phone)\b"),
    ("meizu", r"\bmeizu\b"),
    ("infinix", r"\binfinix\b"),
    ("tecno", r"\btecno\b"),
    ("zte", r"\b(zte|nubia)\b"),
    ("lg", r"\blg\b"),
]

BRAND_PATTERNS = dict(BRAND_RULES)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Название": "title",
        "Цена": "price",
        "URL": "url",
        "Описание": "description",
        "Дата публикации": "date_posted",
        "Продавец": "seller",
        "Адрес": "address",
        "Адрес пользователя": "address_user",
        "Координаты": "coords",
        "Изображения": "images",
        "Поднято": "promoted",
        "Просмотры (всего)": "views_total",
        "Просмотры (сегодня)": "views_today",
        "Телефон": "phone",
        "source_file": "source_file",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=existing)


def normalize_title(text: str) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_brand(text: str) -> str:
    text_l = normalize_title(text).lower()
    for brand, pattern in BRAND_RULES:
        if re.search(pattern, text_l, flags=re.IGNORECASE):
            return brand
    return "unknown"


def extract_title_features(title: str) -> dict:
    title = normalize_title(title)
    title_l = title.lower()
    ram_gb, storage_gb = extract_ram_storage(title)

    return {
        "title": title,
        "title_only": title,
        "brand": infer_brand(title),
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "iphone_gen": extract_iphone_generation(title),
        "is_pro": int(bool(re.search(r"\bpro\b", title_l))),
        "is_max": int(bool(re.search(r"\bmax\b", title_l))),
        "is_mini": int(bool(re.search(r"\bmini\b", title_l))),
        "is_plus": int(bool(re.search(r"\bplus\b", title_l))),
        "title_len": len(title),
    }


def extract_ram_storage(title: str) -> tuple[float | None, float | None]:
    text_l = normalize_title(title).lower()
    if not text_l:
        return None, None

    for pattern in (
        r"(\d{1,2})\s*/\s*(\d{2,4})\s*гб",
        r"(\d{1,2})\s*/\s*(\d{2,4})",
        r"(\d{1,2})\s*gb\s*/\s*(\d{2,4})\s*gb",
    ):
        match = re.search(pattern, text_l)
        if match:
            return float(match.group(1)), float(match.group(2))

    storage_match = re.search(r"(\d{2,4})\s*(?:гб|gb)\b", text_l)
    if storage_match:
        return None, float(storage_match.group(1))

    return None, None


def extract_iphone_generation(title: str) -> float | None:
    text_l = normalize_title(title).lower()
    if not text_l:
        return None
    match = re.search(r"\biphone\s*(\d{1,2})\b", text_l)
    if match:
        return float(match.group(1))
    return None


def classify_condition(text: str) -> str:
    text_l = (text or "").lower()
    if any(w in text_l for w in ["новый", "не распакован", "запечатан", "sealed"]):
        return "new"
    if any(w in text_l for w in ["как новый", "идеальное", "отличное состояние"]):
        return "like_new"
    if any(w in text_l for w in ["б/у", "пользовался", "рабочий", "царапин", "потерт"]):
        return "used"
    if any(w in text_l for w in ["не работает", "на запчасти", "битый", "слом", "нерабоч"]):
        return "broken"
    return "unknown"


def keyword_flag(text: str, keywords: list[str]) -> int:
    text_l = (text or "").lower()
    return int(any(k in text_l for k in keywords))


def build_features(df: pd.DataFrame, *, min_price: int = 500, max_price: int = 200_000) -> pd.DataFrame:
    out = normalize_columns(df.copy())

    title_feats = out["title"].apply(extract_title_features).apply(pd.Series)
    out = pd.concat([out.drop(columns=["title"]), title_feats], axis=1)

    out["description"] = out["description"].fillna("").astype(str)
    out["text"] = (out["title"] + " " + out["description"]).str.strip()

    desc_l = out["description"].str.lower()
    out["condition"] = out["text"].map(classify_condition)
    out["is_promoted"] = (out.get("promoted", pd.Series(dtype=str)).fillna("").astype(str).str.lower() == "да").astype(int)
    out["has_warranty"] = desc_l.map(lambda x: keyword_flag(x, ["гарант", "warranty"]))
    out["has_trade_in"] = desc_l.map(lambda x: keyword_flag(x, ["trade", "обмен", "trade-in"]))
    out["has_box"] = desc_l.map(lambda x: keyword_flag(x, ["коробк", "box"]))
    out["has_charger"] = desc_l.map(lambda x: keyword_flag(x, ["заряд", "charger", "адаптер"]))

    out["photos_count"] = out.get("images", pd.Series(dtype=str)).fillna("").astype(str).map(
        lambda x: len([p for p in x.split(";") if p.strip()])
    )
    out["description_len"] = out["description"].str.len()

    out["date_posted"] = pd.to_datetime(out["date_posted"], errors="coerce")
    ref = out["date_posted"].max()
    out["days_since_posted"] = (ref - out["date_posted"]).dt.days if pd.notna(ref) else None

    addr = out.get("address", pd.Series(dtype=str)).fillna("")
    if "address_user" in out.columns:
        addr = addr.where(addr.astype(str).str.len() > 0, out["address_user"].fillna(""))
    out["district"] = addr.astype(str).map(lambda x: x.split(",")[0].strip() if x else "unknown")

    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["price", "title"])
    out = out[(out["price"] >= min_price) & (out["price"] <= max_price)].reset_index(drop=True)
    return out
