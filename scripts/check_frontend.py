import urllib.request
import re

url = "http://localhost:3000"
try:
    res = urllib.request.urlopen(url)
    html = res.read().decode("utf-8")
    print(f"URL: {url} -> Status: {res.status}, Length: {len(html)}")

    # Check scripts
    scripts = re.findall(r'src="([^"]+)"', html)
    print(f"Scripts count: {len(scripts)}")
    for s in scripts[:5]:
        s_url = f"{url}{s}" if s.startswith("/") else s
        try:
            r = urllib.request.urlopen(s_url)
            print(f"  Script {s} -> {r.status} ({len(r.read())} bytes)")
        except Exception as e:
            print(f"  Script {s} -> FAILED: {e}")

    # Check images
    imgs = re.findall(r'src="([^"]+\.(?:png|jpg|svg))"', html)
    print(f"Images count: {len(imgs)}")
    for img in imgs:
        img_url = f"{url}{img}" if img.startswith("/") else img
        try:
            r = urllib.request.urlopen(img_url)
            print(f"  Image {img} -> {r.status} ({len(r.read())} bytes)")
        except Exception as e:
            print(f"  Image {img} -> FAILED: {e}")

except Exception as e:
    print(f"Failed to connect to {url}: {e}")
