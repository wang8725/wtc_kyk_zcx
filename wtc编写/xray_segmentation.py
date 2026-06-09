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

def enhance_image(img):
    """图像增强：CLAHE对比度增强 + 非锐化掩膜，提升清晰度同时降低干扰"""
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    enhanced = clahe.apply(img)
    # 非锐化掩膜（Unsharp Masking）：高斯模糊后与原图做差，叠加回原图实现锐化
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return sharpened

def enhance_edges(img):
    """梯度增强：突出缺陷边缘，便于阈值分割捕捉"""
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = np.uint8(np.clip(grad, 0, 255))
    return grad

def adaptive_median_filter(img, max_window=7):
    """
    自适应中值滤波：根据局部噪声密度动态调整滤波窗口大小
    相比普通中值滤波，能在去噪的同时更好地保留边缘细节
    """
    h, w = img.shape
    padded = np.pad(img, max_window // 2, mode='reflect')
    result = img.copy().astype(np.float32)

    for i in range(h):
        for j in range(w):
            window_size = 3
            while window_size <= max_window:
                r = window_size // 2
                window = padded[i:i + window_size, j:j + window_size]
                z_min = np.min(window)
                z_max = np.max(window)
                z_med = np.median(window)
                z_xy = padded[i + max_window // 2, j + max_window // 2]

                a1 = z_med - z_min
                a2 = z_med - z_max
                if a1 > 0 and a2 < 0:
                    # z_med 不是脉冲噪声
                    b1 = z_xy - z_min
                    b2 = z_xy - z_max
                    if b1 > 0 and b2 < 0:
                        result[i, j] = z_xy  # 保留原像素
                    else:
                        result[i, j] = z_med  # 用中值替换
                    break
                window_size += 2

            if window_size > max_window:
                result[i, j] = z_med

    return np.uint8(np.clip(result, 0, 255))


def wavelet_haar_2d(img):
    """二维Haar小波变换：分解为LL(近似)、LH(水平)、HL(垂直)、HH(对角)四个子带"""
    h, w = img.shape
    img_f = img.astype(np.float32)

    # 行变换
    row_low = (img_f[:, 0::2] + img_f[:, 1::2]) / np.sqrt(2)
    row_high = (img_f[:, 0::2] - img_f[:, 1::2]) / np.sqrt(2)

    # 列变换
    LL = (row_low[0::2, :] + row_low[1::2, :]) / np.sqrt(2)
    LH = (row_low[0::2, :] - row_low[1::2, :]) / np.sqrt(2)
    HL = (row_high[0::2, :] + row_high[1::2, :]) / np.sqrt(2)
    HH = (row_high[0::2, :] - row_high[1::2, :]) / np.sqrt(2)

    return LL, LH, HL, HH


def wavelet_ihaar_2d(LL, LH, HL, HH):
    """二维Haar小波逆变换：从四个子带重建图像"""
    h2, w2 = LL.shape

    # 逆列变换
    row_low = np.zeros((h2 * 2, w2), dtype=np.float32)
    row_high = np.zeros((h2 * 2, w2), dtype=np.float32)

    row_low[0::2, :] = (LL + LH) / np.sqrt(2)
    row_low[1::2, :] = (LL - LH) / np.sqrt(2)
    row_high[0::2, :] = (HL + HH) / np.sqrt(2)
    row_high[1::2, :] = (HL - HH) / np.sqrt(2)

    # 逆行变换
    img_f = np.zeros((h2 * 2, w2 * 2), dtype=np.float32)
    img_f[:, 0::2] = (row_low + row_high) / np.sqrt(2)
    img_f[:, 1::2] = (row_low - row_high) / np.sqrt(2)

    return img_f


def wavelet_denoise(img, thr_factor=0.4):
    """
    小波变换去噪：对高频子带系数进行软阈值处理
    - 将图像分解为LL(近似)和LH/HL/HH(细节)子带
    - 对细节系数使用自适应软阈值收缩，去除噪声
    - 重建图像，保留结构信息
    """
    h, w = img.shape
    # 确保尺寸为偶数
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    img_cropped = img[:h_even, :w_even]

    LL, LH, HL, HH = wavelet_haar_2d(img_cropped)

    # 使用Donoho通用阈值（去噪）
    sigma = np.median(np.abs(HH - np.median(HH))) / 0.6745
    threshold = thr_factor * sigma * np.sqrt(2 * np.log(max(h_even, w_even)))

    # 软阈值处理
    def soft_threshold(coeff, thr):
        return np.sign(coeff) * np.maximum(np.abs(coeff) - thr, 0)

    LH_d = soft_threshold(LH, threshold)
    HL_d = soft_threshold(HL, threshold)
    HH_d = soft_threshold(HH, threshold)

    denoised = wavelet_ihaar_2d(LL, LH_d, HL_d, HH_d)

    # 恢复到原尺寸
    result = np.zeros_like(img, dtype=np.float32)
    result[:h_even, :w_even] = denoised
    if h_even < h:
        result[h_even:, :] = img[h_even:, :]
    if w_even < w:
        result[:, w_even:] = img[:, w_even:]

    return np.uint8(np.clip(result, 0, 255))


def parallel_filter(img):
    """
    并行滤波：同时进行自适应中值滤波和小波变换去噪，选择性融合
    - 小波变换去噪作为主体，有效去除高斯噪声，保留结构特征
    - 自适应中值滤波仅在椒盐噪声点（邻域内孤立极值）处介入修正
    - 避免中值滤波过度平滑破坏小波去噪保留的细节
    """
    filt_median = adaptive_median_filter(img, max_window=5)
    filt_wavelet = wavelet_denoise(img, thr_factor=0.4)

    # 检测椒盐噪声：像素值与局部中值差异过大的位置
    local_median = cv2.medianBlur(img, 5)
    diff = np.abs(img.astype(np.float32) - local_median.astype(np.float32))
    # 差异超过阈值（暗/亮像素点）判定为椒盐噪声候选
    noise_mask = (diff > 30).astype(np.float32)
    # 对掩膜做高斯平滑，使融合边界自然过渡
    noise_mask = cv2.GaussianBlur(noise_mask, (5, 5), 2.0)

    # 选择性融合：噪声区域用中值修正，其余保留小波结果
    fused = (filt_wavelet.astype(np.float32) * (1 - noise_mask) +
             filt_median.astype(np.float32) * noise_mask)
    fused = np.uint8(np.clip(fused, 0, 255))

    return filt_median, filt_wavelet, fused


def filter_small_components(binary, min_area):
    """连通域分析：过滤面积小于min_area的噪声区域"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255
    return mask

def detect_weld_roi(img):
    """
    自动检测焊缝ROI，排除数字/文本区域。
    原理：文本区域行方向的局部方差显著高于焊缝区域。
    返回掩膜：255=焊缝（需处理），0=文本区域（排除）。
    """
    h, w = img.shape
    window = max(3, h // 30)

    # 行方向局部标准差（纹理复杂度指标）
    local_std = np.array([np.std(img[max(0, i-window):min(h, i+window+1), :])
                          for i in range(h)], dtype=np.float32)

    # 平滑处理
    local_std = cv2.GaussianBlur(local_std.reshape(-1, 1), (max(3, h//20)|1, 1), 2.0).flatten()

    # 自适应阈值：高于均值+0.3倍标准差的区域判定为文本区
    thr = np.mean(local_std) + 0.3 * np.std(local_std)
    text_rows = (local_std > thr).astype(np.uint8)

    # 形态学膨胀：合并相邻文本行
    kernel = np.ones(max(3, h // 25), np.uint8)
    text_band = cv2.dilate(text_rows, kernel).flatten()

    # 只排除顶部和底部的文本带（中间不会出现文本）
    # 找到文本带边界
    top_end = 0
    for i in range(h):
        if text_band[i] == 0:
            top_end = i
            break
    bottom_start = h
    for i in range(h - 1, -1, -1):
        if text_band[i] == 0:
            bottom_start = i + 1
            break

    # 判断文本带是否合理（不超过图像高度的25%，防止误判）
    if top_end > h * 0.25:
        top_end = 0  # 可能是噪声误判，取消顶部排除
    if h - bottom_start > h * 0.25:
        bottom_start = h  # 取消底部排除

    # 只在真正的文本带区域设置为0
    mask = np.full((h, w), 255, dtype=np.uint8)
    if top_end > 0:
        mask[:top_end, :] = 0
    if bottom_start < h:
        mask[bottom_start:, :] = 0

    return mask, top_end, bottom_start


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

def create_segmentation(img,pm,dm,en=None,filt_median=None,filt_wavelet=None,filt_fused=None,roi_bounds=None):
    panels=[("原图",img)]
    # 在原图上标注ROI边界
    img_roi = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if roi_bounds is not None:
        top, bottom = roi_bounds
        if top > 0:
            cv2.line(img_roi, (0, top), (img.shape[1], top), (0, 255, 255), 2)
        if bottom < img.shape[0]:
            cv2.line(img_roi, (0, bottom), (img.shape[1], bottom), (0, 255, 255), 2)
    panels[0] = ("原图(ROI)", img_roi)
    if en is not None:panels.append(("增强图(CLAHE)",en))
    if filt_median is not None:panels.append(("自适应中值滤波",filt_median))
    if filt_wavelet is not None:panels.append(("小波变换去噪",filt_wavelet))
    if filt_fused is not None:panels.append(("并行融合",filt_fused))
    panels+=[("零件掩膜(Otsu)",pm),("缺陷掩膜",dm)]
    return _hstack_panels(panels)
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

def create_comparison(img,gr,ar,filt_median=None,filt_wavelet=None,filt_fused=None):
    dn,dm,tm=img.size*0.00002,img.size*0.005,img.size*0.001
    o=cv2.cvtColor(gr['otsu_bin'],cv2.COLOR_GRAY2BGR)
    tri=cv2.cvtColor(gr['tri_bin'],cv2.COLOR_GRAY2BGR)
    am=cv2.cvtColor(ar['adp_mean_bin'],cv2.COLOR_GRAY2BGR)
    ag=cv2.cvtColor(ar['adp_gauss_bin'],cv2.COLOR_GRAY2BGR)
    panels=[(f"Otsu T={gr['otsu_val']}",o),(f"Triangle T={gr['tri_val']}",tri),(f"自适应均值块{ar['block_size']}",am),(f"自适应高斯块{ar['block_size']}",ag)]
    if filt_median is not None:
        panels.append(("自适应中值滤波",filt_median))
    if filt_wavelet is not None:
        panels.append(("小波变换去噪",filt_wavelet))
    if filt_fused is not None:
        panels.append(("并行融合",filt_fused))
    for im,fts in zip([o,tri,am,ag],[gr['feats_otsu'],gr['feats_tri'],ar['feats_mean'],ar['feats_gauss']]):
        for ft in fts:
            lab,col=classify(ft,dn,dm,tm)
            if lab=='noise':continue
            x,y,bw,bh=ft['bbox_x'],ft['bbox_y'],ft['bbox_w'],ft['bbox_h']
            cv2.rectangle(im,(x,y),(x+bw,y+bh),col,1)
            im=put_cn_text(im,c(lab),(x,max(y-14,2)),11,col)
    return _hstack_panels(panels)

def process_image(n):
    img=read_image(os.path.join(IMG_DIR,n))
    if img is None:print(f'失败:{n}');return
    stem=os.path.splitext(n)[0];h,w=img.shape;allA=h*w
    # --- 图像增强：CLAHE + 非锐化掩膜，提升清晰度 ---
    enhanced=enhance_image(img)
    grad=enhance_edges(img)
    # --- 并行滤波：自适应中值滤波 + 小波变换去噪 ---
    filt_median, filt_wavelet, filt_fused = parallel_filter(enhanced)
    # 融合梯度信息（轻度），增强缺陷边缘供阈值分割使用
    enhanced_grad=cv2.addWeighted(filt_fused,0.85,grad,0.15,0)
    blur=cv2.GaussianBlur(filt_fused,(3,3),0)
    ot,otb=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    tr,trb=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    k5,k7,k3=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    pm=cv2.morphologyEx(otb,cv2.MORPH_CLOSE,k7);pm=cv2.morphologyEx(pm,cv2.MORPH_OPEN,k5)
    pc,_=cv2.findContours(pm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    pc=[i for i in pc if cv2.contourArea(i)>allA*0.005]
    pmc=np.zeros_like(pm);cv2.drawContours(pmc,pc,-1,255,cv2.FILLED)
    # --- 缺陷检测：梯度增强 + 中值滤波去椒盐噪声 ---
    bs=max(11,(min(h,w)//15)|1)
    adg=cv2.adaptiveThreshold(enhanced_grad,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bs,2)
    adgc=cv2.morphologyEx(adg,cv2.MORPH_OPEN,k3)
    adgc=cv2.medianBlur(adgc,3)
    dfm=cv2.bitwise_and(adgc,pmc)
    # --- ROI选取：自动检测并排除数字/文本区域，仅处理焊缝 ---
    roi_mask, top_end, bottom_start = detect_weld_roi(img)
    dfm = cv2.bitwise_and(dfm, roi_mask)  # 排除文本区域
    roi_info = f'ROI: [{top_end}:{bottom_start}]' if top_end > 0 or bottom_start < h else 'ROI: 全图'
    # 连通域过滤：去除小面积噪声
    dfm=filter_small_components(dfm,int(allA*0.00001))
    dfc,_=cv2.findContours(dfm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    dfc=[i for i in dfc if cv2.contourArea(i)>allA*0.00002]
    dfmask=np.zeros_like(dfm);cv2.drawContours(dfmask,dfc,-1,255,cv2.FILLED)
    dn,dm,tm=allA*0.00002,allA*0.005,allA*0.001;dfs=compute_features(dfc,img.shape)
    #全局阈值
    oti,triinv=cv2.bitwise_not(otb),cv2.bitwise_not(trb)
    otc,tric=cv2.morphologyEx(oti,cv2.MORPH_OPEN,k3),cv2.morphologyEx(triinv,cv2.MORPH_OPEN,k3)
    otc,tric=cv2.bitwise_and(otc,roi_mask),cv2.bitwise_and(tric,roi_mask)
    cot,_=cv2.findContours(otc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ctr,_=cv2.findContours(tric,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    gr={'otsu_val':ot,'tri_val':tr,'otsu_bin':otc,'tri_bin':tric,'feats_otsu':compute_features(cot,img.shape),'feats_tri':compute_features(ctr,img.shape)}
    #自适应（对比用）
    adm=cv2.adaptiveThreshold(enhanced_grad,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,bs,2)
    admb=cv2.adaptiveThreshold(enhanced_grad,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bs,2)
    admc,admbc=cv2.morphologyEx(adm,cv2.MORPH_OPEN,k3),cv2.morphologyEx(admb,cv2.MORPH_OPEN,k3)
    admc,admbc=cv2.bitwise_and(admc,roi_mask),cv2.bitwise_and(admbc,roi_mask)
    cam,_=cv2.findContours(admc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cag,_=cv2.findContours(admbc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ar={'block_size':bs,'C':2,'adp_mean_bin':admc,'adp_gauss_bin':admbc,'feats_mean':compute_features(cam,img.shape),'feats_gauss':compute_features(cag,img.shape)}
    #保存5图
    save_image(f'{OUT_DIR}/{stem}_1_分割结果.png',create_segmentation(img,pmc,dfmask,enhanced,filt_median,filt_wavelet,filt_fused,(top_end,bottom_start)))
    save_image(f'{OUT_DIR}/{stem}_2_框选标注.png',create_annotation(img,dfs,dn,dm,tm))
    save_image(f'{OUT_DIR}/{stem}_3_缺陷分类.png',create_classification(img,pmc,dfs,dn,dm,tm))
    save_csv(dfs,f'{OUT_DIR}/{stem}_4_特征统计.csv',dn,dm,tm)
    save_image(f'{OUT_DIR}/{stem}_5_算法对比.png',create_comparison(img,gr,ar,filt_median,filt_wavelet,filt_fused))
    #txt统计
    cntdic={'裂纹':0,'气孔':0,'不规则缺陷':0,'数字':0,'正常纹理':0,'不规则纹理':0,'噪声':0}
    for f in dfs:cntdic[c(classify(f,dn,dm,tm)[0])]+=1
    txt=[f'===={n}====',f'Otsu:{ot} Tri:{tr} 块:{bs} C:2 阈值:噪声<{dn:.0f}小缺陷<{dm:.0f}大区≥{tm:.0f} | {roi_info} | CLAHE+锐化+自适应中值滤波+小波变换并行去噪',f'总数{len(dfs)} {cntdic}']
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