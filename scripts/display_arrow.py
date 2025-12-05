from datasets import load_from_disk
from pprint import pprint

ds = load_from_disk("data/msrvtt_hf/msrvtt_train")  

print("Features:", ds.features)
print("Length:", len(ds))

ex = ds[0]
pprint(ex)
print("Caption:", ex["caption"])
print("Video ID:", ex["video_id"])
print("Num frames:", len(ex["frames"]))
print("Frame type:", type(ex["frames"][0]))
