
def load_hyp_txt(file="mergehyp.txt"):
    with open(file,'r') as fid:
        lines = fid.readlines()
    return lines

def write_ctm():
    fid = open("hyp.ctm",'w')
    lines = load_hyp_txt()
    format = "%s %s 1 %d %d %s\n"
    for l in lines:
        l = l.strip().split()
        k,words = l[0],l[1:]
        if ('-a' in k):
            channel = 'A'
        else:
            channel = 'B'
        t = 0
        for word in words:
            fid.write(format%(k[0:7], channel, t, 1, word))
            t += 1
    fid.close()

if __name__=='__main__':
    write_ctm()
