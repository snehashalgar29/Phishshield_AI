import re
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    t = text.replace('\n',' ').replace('\r',' ').strip()
    t = re.sub(r'\s+', ' ', t)
    return t
