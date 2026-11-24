import os,glob,numpy as np
from PIL import Image
def th(p):  # aspect-free 32x32 grey thumbnail
    return np.asarray(Image.open(p).convert("L").resize((32,32),Image.BILINEAR),float)
kpp=glob.glob("keypoint_detection/images/train/*")
K=np.stack([th(p) for p in kpp]); kst=[os.path.splitext(os.path.basename(p))[0] for p in kpp]
for cs in ["test","val"]:
    files=[p for c in ["Leaned","Straight","Rejected"] for p in glob.glob(f"classification/{cs}/{c}/*")]
    T=np.stack([th(p) for p in files])
    D=np.abs(T[:,None]-K[None]).mean((2,3))
    j=D.argmin(1); d=D.min(1)
    st=[os.path.splitext(os.path.basename(p))[0] for p in files]
    same_name=np.array([s in set(kst) for s in st])
    near=d<3
    print(f"cls {cs}: {near.sum()} images have a near-identical image (thumb diff<3/255) in kp-train; "
          f"of these same file name: {(near&same_name).sum()}, different name: {(near&~same_name).sum()}; "
          f"same-name but NOT near-identical: {(same_name&~near).sum()}")
    bad=[(st[i],kst[j[i]],round(d[i],1)) for i in np.where(same_name&~near)[0]][:5]; print("  examples same-name-not-identical:",bad)
