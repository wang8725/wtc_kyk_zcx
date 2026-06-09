import cv2, numpy as np, os, csv
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR, OUT_DIR = os.path.join(BASE_DIR, '图像', 'image'), os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)
FONT_PATH = r'C:\Windows\Fonts\simhei.ttf'
CN = {'noise':'噪声','crack':'裂纹','porosity':'气孔','irregular_defect':'不规则缺陷','text_digit':'数字','large_region':'大区域','texture_irregular':'不规则纹理','normal_texture':'正常纹理','unknown':'未知'}
COLOR_MAP = {'crack':(0,0,255),'porosity':(255,0,255),'irregular_defect':(0,165,255),'text_digit':(255,0,0),'normal_texture':(0,255,0),'texture_irregular':(255,255,0)}

def read_image(p):
    d = np.fromfile(p,np.uint8)
    im = cv2.imdecode(d,cv2.IMREAD_GRAYSCALE)
    if im is None:im=cv2.cvtColor(cv2.imdecode(d,cv2.IMREAD_COLOR),cv2.COLOR_BGR2GRAY)
    return im
def save_image(p,im):
    _,e=cv2.imencode(os.path.splitext(p)[1],im if len(im.shape)==2 else cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
    e.tofile(p)
def put_cn_text(im,txt,pos,sz,c):
    pi=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pi).text(pos,txt,font=ImageFont.truetype(FONT_PATH,sz),fill=(c[2],c[1],c[0]))
    return cv2.cvtColor(np.array(pi),cv2.COLOR_RGB2BGR)
c=lambda k:CN.get(k,k)

def compute_features(cts,sh):
    fs,h,w,areaAll=[],sh[0],sh[1],sh[0]*sh[1]
    for idx,cnt in enumerate(cts):
        ar,pr=cv2.contourArea(cnt),cv2.arcLength(cnt,True)
        if pr==0:continue
        x,y,bw,bh=cv2.boundingRect(cnt)
        rectAr=bw*bh
        circ=4*np.pi*ar/(pr**2)
        m=cv2.moments(cnt)
        cx=m['m10']/m['m00'] if m['m00'] else x+bw/2
        cy=m['m01']/m['m00'] if m['m00'] else y+bh/2
        sol=ar/cv2.contourArea(cv2.convexHull(cnt)) if cv2.contourArea(cv2.convexHull(cnt)) else 0
        fs.append({'id':idx+1,'area':ar,'area_pct':100*ar/areaAll,'perimeter':pr,'circularity':circ,'bbox_x':x,'bbox_y':y,'bbox_w':bw,'bbox_h':bh,'aspect_ratio':bw/bh if bh else 0,'extent':ar/rectAr if rectAr else 0,'solidity':sol,'centroid_x':cx,'centroid_y':cy})
    return fs

def classify(f,dn,dm,tm):
    a,cir=f['area'],f['circularity']
    if a<dn:return 'noise',(0,0,0)
    if a<dm:
        if cir<0.3:return 'crack',COLOR_MAP['crack']
        if cir<0.7:return 'irregular_defect',COLOR_MAP['irregular_defect']
        return 'porosity',COLOR_MAP['porosity']
    if a>=tm:return ('text_digit',COLOR_MAP['text_digit']) if cir<0.3 else ('large_region',COLOR_MAP['normal_texture'])
    return ('texture_irregular',COLOR_MAP['texture_irregular']) if cir<0.5 else ('normal_texture',COLOR_MAP['normal_texture'])

def _hstack_panels(pns,th=20):
    maxh=max(i.shape[0] for _,i in pns)+th
    can=np.zeros((maxh,sum(i.shape[1] for _,i in pns),3),np.uint8)
    x=0
    for tit,img in pns:
        pan=np.zeros((maxh,img.shape[1],3),np.uint8)
        pan[th:th+img.shape[0],:]=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) if len(img.shape)==2 else img
        pan=put_cn_text(pan,tit,(5,2),14,(0,255,255))
        can[:,x:x+img.shape[1]]=pan;x+=img.shape[1]
    return can

def create_segmentation(img,pm,dm):return _hstack_panels([("原图",img),("零件掩膜(Otsu)",pm),("缺陷掩膜",dm)])
def create_annotation(img,fs,dn,dm,tm):
    h,w=img.shape;leg=22
    ann=np.zeros((h+leg,w,3),np.uint8);ann[leg:]=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR);ann[:leg]=(40,40,40)
    for f in fs:
        lab,col=classify(f,dn,dm,tm)
        if lab=='noise':continue
        x,y,bw,bh=f['bbox_x'],f['bbox_y'],f['bbox_w'],f['bbox_h']
        cv2.rectangle(ann,(x,y+leg),(x+bw,y+bh+leg),col,1)
    legds=[("裂纹",COLOR_MAP['crack']),("气孔",COLOR_MAP['porosity']),("不规则缺陷",COLOR_MAP['irregular_defect']),("数字/文本",COLOR_MAP['text_digit']),("正常纹理",COLOR_MAP['normal_texture'])]
    for i,(n,cl) in enumerate(legds):
        lx=5+i*135;cv2.rectangle(ann,(lx,5),(lx+12,16),cl,-1)
        ann=put_cn_text(ann,n,(lx+16,1),12,cl)
    return _hstack_panels([("自动框选标注",ann)])

def create_classification(img,pm,fs,dn,dm,tm):
    h,w=img.shape;cls=np.full((h,w,3),50,np.uint8)
    brd=cv2.Canny(pm,100,200);cls[brd>0]=COLOR_MAP['normal_texture']
    for f in fs:
        lab,col=classify(f,dn,dm,tm)
        if lab=='noise':continue
        x,y,bw,bh=f['bbox_x'],f['bbox_y'],f['bbox_w'],f['bbox_h']
        cls[y:y+bh,x:x+bw]=COLOR_MAP.get(lab,(128,128,128))
    legds=[("裂纹",COLOR_MAP['crack']),("气孔",COLOR_MAP['porosity']),("不规则缺陷",COLOR_MAP['irregular_defect']),("数字/文本",COLOR_MAP['text_digit']),("正常纹理",COLOR_MAP['normal_texture']),("零件边界",COLOR_MAP['normal_texture'])]
    for i,(n,cl) in enumerate(legds):
        lx=5+i*120;cv2.rectangle(cls,(lx,2),(lx+12,14),cl,-1)
        cls=put_cn_text(cls,n,(lx+16,0),11,cl)
    return _hstack_panels([("缺陷分类",cls)])

def save_csv(fs,p,dn,dm,tm):
    with open(p,'w',newline='',encoding='utf-8-sig') as f:
        wr=csv.writer(f)
        wr.writerow(['ID','类型','面积','面积%','周长','圆形度','长宽比','Extent','Solidity','BBOX_X','BBOX_Y','BBOX_W','BBOX_H','质心X','质心Y'])
        for ft in fs:
            lab,_=classify(ft,dn,dm,tm)
            wr.writerow([ft['id'],c(lab),f"{ft['area']:.1f}",f"{ft['area_pct']:.3f}",f"{ft['perimeter']:.1f}",f"{ft['circularity']:.3f}",f"{ft['aspect_ratio']:.3f}",f"{ft['extent']:.3f}",f"{ft['solidity']:.3f}",ft['bbox_x'],ft['bbox_y'],ft['bbox_w'],ft['bbox_h'],f"{ft['centroid_x']:.1f}",f"{ft['centroid_y']:.1f}"])

def create_comparison(img,gr,ar):
    dn,dm,tm=img.size*0.00002,img.size*0.005,img.size*0.001
    o=cv2.cvtColor(gr['otsu_bin'],cv2.COLOR_GRAY2BGR)
    tri=cv2.cvtColor(gr['tri_bin'],cv2.COLOR_GRAY2BGR)
    am=cv2.cvtColor(ar['adp_mean_bin'],cv2.COLOR_GRAY2BGR)
    ag=cv2.cvtColor(ar['adp_gauss_bin'],cv2.COLOR_GRAY2BGR)
    for im,fts in zip([o,tri,am,ag],[gr['feats_otsu'],gr['feats_tri'],ar['feats_mean'],ar['feats_gauss']]):
        for ft in fts:
            lab,col=classify(ft,dn,dm,tm)
            if lab=='noise':continue
            x,y,bw,bh=ft['bbox_x'],ft['bbox_y'],ft['bbox_w'],ft['bbox_h']
            cv2.rectangle(im,(x,y),(x+bw,y+bh),col,1)
            im=put_cn_text(im,c(lab),(x,max(y-14,2)),11,col)
    return _hstack_panels([(f"Otsu T={gr['otsu_val']}",o),(f"Triangle T={gr['tri_val']}",tri),(f"自适应均值块{ar['block_size']}",am),(f"自适应高斯块{ar['block_size']}",ag)])

def process_image(n):
    img=read_image(os.path.join(IMG_DIR,n))
    if img is None:print(f'失败:{n}');return
    stem=os.path.splitext(n)[0];h,w=img.shape;allA=h*w
    blur=cv2.GaussianBlur(img,(5,5),0)
    ot,otb=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    tr,trb=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    k5,k7,k3=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    pm=cv2.morphologyEx(otb,cv2.MORPH_CLOSE,k7);pm=cv2.morphologyEx(pm,cv2.MORPH_OPEN,k5)
    pc,_=cv2.findContours(pm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    pc=[i for i in pc if cv2.contourArea(i)>allA*0.005]
    pmc=np.zeros_like(pm);cv2.drawContours(pmc,pc,-1,255,cv2.FILLED)
    bs=max(11,(min(h,w)//15)|1)
    adg=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bs,3)
    adgc=cv2.morphologyEx(adg,cv2.MORPH_OPEN,k3)
    dfm=cv2.bitwise_and(adgc,pmc)
    dfc,_=cv2.findContours(dfm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    dfc=[i for i in dfc if cv2.contourArea(i)>allA*0.00002]
    dfmask=np.zeros_like(dfm);cv2.drawContours(dfmask,dfc,-1,255,cv2.FILLED)
    dn,dm,tm=allA*0.00002,allA*0.005,allA*0.001;dfs=compute_features(dfc,img.shape)
    #全局阈值
    oti,triinv=cv2.bitwise_not(otb),cv2.bitwise_not(trb)
    otc,tric=cv2.morphologyEx(oti,cv2.MORPH_OPEN,k3),cv2.morphologyEx(triinv,cv2.MORPH_OPEN,k3)
    cot,_=cv2.findContours(otc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ctr,_=cv2.findContours(tric,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    gr={'otsu_val':ot,'tri_val':tr,'otsu_bin':otc,'tri_bin':tric,'feats_otsu':compute_features(cot,img.shape),'feats_tri':compute_features(ctr,img.shape)}
    #自适应
    adm=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,bs,3)
    admb=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bs,3)
    admc,admbc=cv2.morphologyEx(adm,cv2.MORPH_OPEN,k3),cv2.morphologyEx(admb,cv2.MORPH_OPEN,k3)
    cam,_=cv2.findContours(admc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cag,_=cv2.findContours(admbc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ar={'block_size':bs,'C':3,'adp_mean_bin':admc,'adp_gauss_bin':admbc,'feats_mean':compute_features(cam,img.shape),'feats_gauss':compute_features(cag,img.shape)}
    #保存5图
    save_image(f'{OUT_DIR}/{stem}_1_分割结果.png',create_segmentation(img,pmc,dfmask))
    save_image(f'{OUT_DIR}/{stem}_2_框选标注.png',create_annotation(img,dfs,dn,dm,tm))
    save_image(f'{OUT_DIR}/{stem}_3_缺陷分类.png',create_classification(img,pmc,dfs,dn,dm,tm))
    save_csv(dfs,f'{OUT_DIR}/{stem}_4_特征统计.csv',dn,dm,tm)
    save_image(f'{OUT_DIR}/{stem}_5_算法对比.png',create_comparison(img,gr,ar))
    #txt统计
    cntdic={'裂纹':0,'气孔':0,'不规则缺陷':0,'数字':0,'正常纹理':0,'不规则纹理':0,'噪声':0}
    for f in dfs:cntdic[c(classify(f,dn,dm,tm)[0])]+=1
    txt=[f'===={n}====',f'Otsu:{ot} Tri:{tr} 块:{bs} 阈值:噪声<{dn:.0f}小缺陷<{dm:.0f}大区≥{tm:.0f}',f'总数{len(dfs)} {cntdic}']
    with open(f'{OUT_DIR}/{stem}_4_特征统计.txt','w',encoding='utf-8')as f:f.write('\n'.join(txt))
    print(f'{n}完成')
    return {'global':gr,'adaptive':ar,'defect_feats':dfs}

def main():
    imgs=sorted([i for i in os.listdir(IMG_DIR) if i.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif'))])
    allres={}
    for n in imgs:
        r=process_image(n)
        if r:allres[n]=r
    print('全部处理完毕')

if __name__ == '__main__':main()