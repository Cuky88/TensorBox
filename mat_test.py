import scipy.io
mat = scipy.io.loadmat('/home/cuky/Devel/data_link/COD20K/info/00001_annotations.mat')
print(len(mat['CarAnnot']))

for an in mat['CarAnnot']:
    print(an[0])
    break

for ob in mat['pp_original']:
    print(ob)
    #break