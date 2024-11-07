import unicodedata
import string


def import_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return unicodedata.normalize("NFC", text)


# vocabulary = string.printable + "ñÑáÁéÉíÍóÓúÚ¿¡üÜ–«»‘’" y otros caracteres...
def create_vocabulary(text):
    additional = set(text) - set(string.printable)
    vocabulary = string.printable + "".join(additional)
    return vocabulary
