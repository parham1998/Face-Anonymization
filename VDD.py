from vdna import VDNAProcessor, EMD

vdna_proc = VDNAProcessor()

vdna1 = vdna_proc.make_vdna(source="assets/datasets/FFHQ/all", num_workers=0)

vdna2 = vdna_proc.make_vdna(source="results/images", num_workers=0)

emd = EMD(vdna1, vdna2)
print(emd)





