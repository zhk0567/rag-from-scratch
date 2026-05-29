"""抓取网页正文。"""

import re

import requests


def fetch_url_text(url: str, timeout: int = 15) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ImportError("需要安装: pip install beautifulsoup4 requests") from e

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "rag-from-scratch/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) < 50:
        raise ValueError("页面正文过短，可能抓取失败")
    return text
