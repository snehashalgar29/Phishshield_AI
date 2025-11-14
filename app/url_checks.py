import re, math, requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup

SUSPICIOUS_KEYWORDS = ['verify','account','suspend','urgent','confirm','login','password','reset','click','bank']

def domain_entropy(domain: str) -> float:
    if not domain: return 0.0
    freq = {}
    for ch in domain:
        freq[ch] = freq.get(ch,0) + 1
    H = 0.0
    l = len(domain)
    for v in freq.values():
        p = v / l
        H -= p * math.log2(p)
    return H

def analyze_url(url: str):
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    signals = {}
    signals['has_ip'] = bool(re.match(r'^\\d+\\.\\d+\\.\\d+\\.\\d+$', hostname))
    signals['hostname_length'] = len(hostname)
    signals['domain_entropy'] = domain_entropy(hostname)
    signals['count_dots'] = hostname.count('.')
    lower = url.lower()
    signals['suspicious_tokens'] = sum(1 for k in SUSPICIOUS_KEYWORDS if k in lower)
    signals['page_contains_login_form'] = False
    try:
        resp = requests.get(url, timeout=3)
        if 'text/html' in resp.headers.get('Content-Type',''):
            soup = BeautifulSoup(resp.text,'html.parser')
            if soup.find('input', {'type':'password'}):
                signals['page_contains_login_form'] = True
            signals['page_title'] = soup.title.string if soup.title else ''
    except Exception:
        pass
    # simple scoring
    score = 0
    score += min(40, signals['suspicious_tokens'] * 10)
    score += min(30, int(signals['domain_entropy'] * 5))
    if signals['has_ip']: score += 20
    if signals['count_dots'] >= 4: score += 10
    if signals['page_contains_login_form']: score += 20
    score = min(100, score)
    classification = 'malicious' if score >= 70 else 'suspicious' if score >= 40 else 'safe'
    return {'url': url, 'score': score, 'classification': classification, 'signals': signals}
