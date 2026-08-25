import argparse, glob, hashlib, json, os, re
from PIL import Image
from datasets import Dataset, concatenate_datasets

KEEP = {"k12","multimath","mathv360k","DaTikZ_img2code_train","mathpix"}
PROMPT = "Write the Python code that draws this figure."
SIZE, MIN_CODE, MAX_CODE = 384, 50, 4000
FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)

def to_rgb(img):
    if img.mode in ("RGBA","LA") or (img.mode=="P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")

def letterbox(img):
    img = img.copy()
    img.thumbnail((SIZE,SIZE), Image.LANCZOS)
    c = Image.new("RGB",(SIZE,SIZE),"white")
    c.paste(img, ((SIZE-img.width)//2,(SIZE-img.height)//2))
    return c

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60000)
ap.add_argument("--arrow_dir", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--no_filter", action="store_true")
a = ap.parse_args()

out_dir = os.path.abspath(a.out_dir)
img_dir = os.path.join(out_dir, "imgcode")
os.makedirs(img_dir, exist_ok=True)
print("cwd   :", os.getcwd())
print("images:", img_dir)

files = sorted(glob.glob(os.path.join(a.arrow_dir,"**","*.arrow"), recursive=True))
if not files: raise SystemExit("no .arrow files under " + a.arrow_dir)
print("loading", len(files), "shard(s)")
ds = concatenate_datasets([Dataset.from_file(f) for f in files])
print("rows:", len(ds))

seen_id, seen_h, out = set(), set(), []
skip = {"source":0,"dup_id":0,"no_fence":0,"len":0,"dup_img":0,"error":0}

for i, ex in enumerate(ds):
    if len(out) >= a.n: break
    if not a.no_filter and ex.get("source") not in KEEP: skip["source"]+=1; continue
    if ex["id"] in seen_id: skip["dup_id"]+=1; continue
    seen_id.add(ex["id"])
    m = FENCE.search(ex.get("text") or "")
    if not m: skip["no_fence"]+=1; continue
    code = m.group(1).strip()
    if not (MIN_CODE < len(code) < MAX_CODE): skip["len"]+=1; continue
    try:
        img = to_rgb(ex["image"])
        h = hashlib.md5(img.tobytes()).hexdigest()
        if h in seen_h: skip["dup_img"]+=1; continue
        seen_h.add(h)
        p = os.path.join(img_dir, f"{ex['id']:08d}.png")
        letterbox(img).save(p)
    except Exception as e:
        skip["error"]+=1
        if skip["error"]<5: print("row",i,type(e).__name__,e)
        continue
    out.append({"image":p.replace("\\","/"), "instruction":PROMPT,
                "output":code, "align_source":f"imgcode:{ex['source']}"})
    if len(out)%500==0: print("kept",len(out),"scanned",i+1)

jp = os.path.join(out_dir,"imgcode_norm.json")
json.dump(out, open(jp,"w",encoding="utf-8"), ensure_ascii=False)
print("\nkept",len(out),"skipped",skip)
print("json  ->",jp)
print("images->",img_dir,len(os.listdir(img_dir)),"files")
if out:
    from collections import Counter
    print(Counter(r["align_source"] for r in out))
    print("sample:", out[0]["image"], "|", out[0]["output"][:120])
